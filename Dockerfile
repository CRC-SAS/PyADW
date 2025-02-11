
##########################
## Set GLOBAL arguments ##
##########################

# Set python version
ARG PYTHON_VERSION=3.13

# Set image variant
ARG IMG_VARIANT="-slim"

# Set PyADW installation folder
ARG APP_HOME="/opt/PyADW"

# Set user name and id
ARG USR_NAME="pyadw"
ARG USER_UID="1000"

# Set group name and id
ARG GRP_NAME="pyadw"
ARG USER_GID="1000"

# Set users passwords
ARG ROOT_PWD="root"
ARG USER_PWD=${USR_NAME}

# Set Pycharm version
ARG PYCHARM_VERSION="2024.3.2"



######################################
## Stage 1: Install Python packages ##
######################################

# Create image
FROM python:${PYTHON_VERSION}${IMG_VARIANT} AS py_builder

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Set python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip
# Copy dependencies from build context
COPY requirements.txt requirements.txt
# Install Python dependencies (ver: https://stackoverflow.com/a/17311033/5076110)
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal && export C_INCLUDE_PATH=/usr/include/gdal && \
    python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt



###############################################
## Stage 2: Copy Python installation folders ##
###############################################

# Create image
FROM python:${PYTHON_VERSION}${IMG_VARIANT} AS py_final

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install python dependencies from py_builder
COPY --from=py_builder /usr/src/app/wheels /wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache /wheels/* && \
    rm -rf /wheels



#################################
## Stage 3: Create PyADW image ##
#################################

# Create image
FROM py_final AS app_builder

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Renew ARGs
ARG APP_HOME

# Create APP_HOME folder
RUN mkdir -p ${APP_HOME}

# Copy project
COPY . ${APP_HOME}

# Save Git commit hash of this build into ${APP_HOME}/repo_version.
# https://github.com/docker/hub-feedback/issues/600#issuecomment-475941394
# https://docs.docker.com/build/building/context/#keep-git-directory
COPY ./.git /tmp/git
RUN export head=$(cat /tmp/git/HEAD | cut -d' ' -f2) && \
    if echo "${head}" | grep -q "refs/heads"; then \
    export hash=$(cat /tmp/git/${head}); else export hash=${head}; fi && \
    echo "${hash}" > ${APP_HOME}/repo_version && rm -rf /tmp/git

# Set permissions of app files
RUN chmod -R ug+rw,o+r,o-w ${APP_HOME}



###########################################
## Stage 4: Install management packages  ##
###########################################

# Create image
FROM app_builder AS app_mgmt

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        # install Tini (https://github.com/krallin/tini#using-tini)
        tini \
        # to see process with pid 1
        htop procps \
        # to allow edit files
        vim \
        # to show progress through pipelines
        pv && \
    rm -rf /var/lib/apt/lists/*

# Add Tini (https://github.com/krallin/tini#using-tini)
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]



######################################
## Stage 5: Setup PyADW core image  ##
######################################

# Create image
FROM app_mgmt AS app-core

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Renew ARGs
ARG APP_HOME

# Add Tini (https://github.com/krallin/tini#using-tini)
ENTRYPOINT [ "/usr/bin/tini", "-g", "--" ]

# Run your program under Tini (https://github.com/krallin/tini#using-tini)
CMD [ "python", "--version" ]
# or docker run your-image /your/program ...

# Set work directory
WORKDIR ${APP_HOME}



###################################
## Stage 6: Create non-root user ##
###################################

# Create image
FROM app-core AS app_nonroot_builder

# Renew ARGs
ARG USR_NAME
ARG USER_UID
ARG GRP_NAME
ARG USER_GID
ARG ROOT_PWD
ARG USER_PWD

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        # to run sudo
        sudo && \
    rm -rf /var/lib/apt/lists/*

# Modify root password
RUN echo "root:${ROOT_PWD}" | chpasswd

# Create a non-root user, so the container can run as non-root
# OBS: the UID and GID must be the same as the user that own the
# input and the output volumes, so there isn't perms problems!!
# Se recomienda crear usuarios en el contendor de esta manera,
# ver: https://nickjanetakis.com/blog/running-docker-containers-as-a-non-root-user-with-a-custom-uid-and-gid
# Se agregar --no-log-init para prevenir un problema de seguridad,
# ver: https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
RUN groupadd --gid ${USER_GID} ${GRP_NAME}
RUN useradd --no-log-init --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash \
    --comment "Non-root User Account" --create-home ${USR_NAME}

# Modify the password of non-root user
RUN echo "${USR_NAME}:${USER_PWD}" | chpasswd

# Add non-root user to sudoers and to adm group
# The adm group was added to allow non-root user to see logs
RUN usermod -aG sudo ${USR_NAME} && \
    usermod -aG adm ${USR_NAME}



################################################
## Stage 7: Install Pycharm (for development) ##
################################################

# Create image
FROM app_nonroot_builder AS app-pycharm

# Become root
USER root

# Renew ARGs
ARG APP_HOME
ARG USR_NAME
ARG GRP_NAME

# Update apt cache and install useful packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        curl wget git build-essential python3-tk

# Renew ARGs
ARG PYCHARM_VERSION

# Download Pycharm IDE
RUN wget https://download.jetbrains.com/python/pycharm-community-${PYCHARM_VERSION}.tar.gz -P /tmp/

# Install packages required to run PyCharm IDE
RUN count=$(ls /tmp/pycharm-*.tar.gz | wc -l) && [ ${count} = 1 ] \
    && apt-get -y -qq --no-install-recommends install \
        # Without this packages, PyCharm don't start
        libxrender1 libxtst6 libxi6 libfreetype6 fontconfig \
        # Without this packages, PyCharm start, but reports that they are missing
        libatk1.0-0 libatk-bridge2.0-0 libdrm-dev libxkbcommon-dev libdbus-1-3 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr-dev libgbm1 libasound2 \
        libcups2 libatspi2.0-0 libxshmfence1 \
        # Without this packages, PyCharm start, but shows errors when running
        procps libsecret-1-0 gnome-keyring libxss1 libxext6 firefox-esr dbus-x11 \
        libcanberra-gtk-module libcanberra-gtk3-module \
        #libnss3 libxext-dev libnspr4 \
    || :  # para entender porque :, ver https://stackoverflow.com/a/49348392/5076110

# Install PyCharm IDE
RUN count=$(ls /tmp/pycharm-*.tar.gz | wc -l) && [ ${count} = 1 ] \
    && mkdir /opt/pycharm \
    && tar xzf /tmp/pycharm-*.tar.gz -C /opt/pycharm --strip-components 1 \
    && chown -R ${USR_NAME}:${GRP_NAME} /opt/pycharm \
    || :  # para entender porque :, ver https://stackoverflow.com/a/49348392/5076110

# Setup Pycharm IDE
RUN mkdir -p ${APP_HOME}/.idea \
    && chown -R ${USR_NAME}:${GRP_NAME} ${APP_HOME}/.idea

# Renew ARGs
ARG PYTHON_VERSION

# Pycharm espera que los paquetes python estén en dist-packages, pero están en site-packages.
# Esto es así porque python no se instaló usando apt o apt-get, y cuando esto ocurre, la carpeta
# en la que se instalan los paquetes es site-packages y no dist-packages.
RUN mkdir -p /usr/local/lib/python${PYTHON_VERSION}/dist-packages \
    && ln -s /usr/local/lib/python${PYTHON_VERSION}/site-packages/* \
             /usr/local/lib/python${PYTHON_VERSION}/dist-packages/

# Change to non-root user
USER ${USR_NAME}

# Set work directory
WORKDIR ${APP_HOME}

# Run pycharm under Tini (https://github.com/krallin/tini#using-tini)
CMD ["/opt/pycharm/bin/pycharm", "-Dide.browser.jcef.enabled=true"]
# or docker run your-image /your/program ...


#
# Ejecución de PyCharm:
#
# 1- docker build --force-rm \
#      --target app-pycharm \
#      --tag pyadw-pycharm:latest \
#      --build-arg USER_UID=$(stat -c "%u" .) \
#      --build-arg USER_GID=$(stat -c "%g" .) \
#      --file Dockerfile .
#
# 2- docker volume create pyadw-home
#
# 3- docker run --tty --interactive --rm --name chwon-pyadw-home \
#      --mount type=volume,source=pyadw-home,target=/home/pyadw \
#      --env uUID=$(stat -c "%u" .) --env uGID=$(stat -c "%g" .) \
#      debian:stable-slim bash -c "chown \$uUID:\$uGID -Rv /home/pyadw"
#
# When using wayland (e.g., Ubuntu 24.04). To run PyCharm using the host display, you must run
# "xhost +" before "docker run", and "xhost -" after it (https://unix.stackexchange.com/q/593411)
# or "xhost +SI:localuser:$(id -un)" instead (https://unix.stackexchange.com/a/359244)
#
# 4- xhost +SI:localuser:"$(id -un)"
#
# 5- docker run --tty --interactive --rm \
#      --name pyadw-pycharm \
#      --env DISPLAY="${DISPLAY}" \
#      --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
#      --mount type=volume,source=pyadw-home,target=/home/pyadw \
#      --mount type=bind,source=$(pwd),target=/opt/PyADW \
#      --detach pyadw-pycharm:latest
#

