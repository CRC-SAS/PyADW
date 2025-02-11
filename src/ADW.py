import numpy as np

class ADW:

    def __init__(self,
                 obslon,
                 obslat,
                 obs,
                 extent,
                 res=1,
                 cdd=1e3,
                 m=4,
                 nmin=3,
                 nmax=4
                 ) -> None:

        self.obslon = obslon
        self.obslat = obslat
        self.obs = obs
        self.cdd = cdd
        self.m = m
        self.nmax = nmax
        self.nmin = nmin

        self.xrange = np.arange(extent[0], extent[1], res if extent[0] < extent[1] else res * -1)
        self.yrange = np.arange(extent[2], extent[3], res if extent[2] < extent[3] else res * -1)
        self.xgrd, self.ygrd = np.meshgrid(
            self.xrange,
            self.yrange
        )
        self.lons = self.xgrd.flatten()
        self.lats = self.ygrd.flatten()

        self.grid = np.zeros((self.yrange.shape[0], self.xrange.shape[0]))

    def interpolate(self):
        for idx, lon in np.ndenumerate(self.lons):

            # computar distancias del punto seleccionado a los datos observados
            distance = self.distance_gcd(lon, self.lats[idx], self.obslon, self.obslat)

            # crear stack para manejar los datos como en un dataframe
            ds_tmp = np.stack([self.obs, distance, self.obslon, self.obslat], axis=1)

            # seleccionar solo datos observados más cercanos que self.cdd
            dx = ds_tmp[np.argwhere(distance < self.cdd).reshape(-1)][:]

            # contar datos observados más cercanos que self.cdd
            npts = len(dx)

            # definir valor por defecto para el punto interpolado!!
            value = np.nan

            if npts > self.nmax:
                sort_dist_idx = np.argsort(dx[::,1])
                dx = dx[::][sort_dist_idx]
                dx = dx[0:self.nmax]
                w = self.weight_cal(dx[::,1], lon, self.lats[idx], dx[::,2], dx[::,3])
                value = np.sum(dx[::,0] * w) / np.sum(w)

            elif self.nmin <= npts <= self.nmax:
                w = self.weight_cal(
                    dx[::,1], lon, self.lats[idx],
                    dx[::,2], dx[::,3]
                )
                value = np.sum(dx[::,0] * w) / np.sum(w)

            # identificar ubicación del valor interpolado en self.grid
            lon_idx = np.argwhere(self.xrange == lon)
            lat_idx = np.argwhere(self.yrange == self.lats[idx])

            # actualizar self.grid
            self.grid[lat_idx, lon_idx] = value

    def distance_gcd(self, lon1, lat1, lon2, lat2):
        radius = 6371.01  # Earth radius, unit: km

        rlon1 = self.deg2rad(lon1)
        rlat1 = self.deg2rad(lat1)
        rlon2 = self.deg2rad(lon2)
        rlat2 = self.deg2rad(lat2)

        dlon = np.abs(rlon1 - rlon2)
        dlat = np.abs(rlat1 - rlat2)

        a = np.sin(dlat / 2.0) ** 2.0 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2.0
        cc = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return radius * cc

    @staticmethod
    def deg2rad(degree):
        return degree * np.pi / 180.0

    def weight_cal(self, stn_distances, grd_lon, grd_lat, stn_lon, stn_lat):
        r = np.exp(-stn_distances / self.cdd)
        f = r ** self.m
        theta = self.bearing_r(grd_lon, grd_lat, stn_lon, stn_lat)
        npts = len(stn_distances)
        alpha = np.zeros(npts)

        for k in range(npts):
            theta_w = np.delete(theta, k)
            diff_theta = theta_w - theta[k]
            f_w = np.delete(f, k)
            alpha[k] = np.nansum(f_w * (1 - np.cos(diff_theta))) / np.nansum(f_w)

        w = f * (1 + alpha)
        return w

    def bearing_r(self, lon1, lat1, lon2, lat2):
        rlon1 = self.deg2rad(lon1)
        rlat1 = self.deg2rad(lat1)
        rlon2 = self.deg2rad(lon2)
        rlat2 = self.deg2rad(lat2)

        y = np.sin(rlon2 - rlon1) * np.cos(rlat2)
        x = np.cos(rlat1) * np.sin(rlat2) - np.sin(rlat1) * np.cos(rlat2) * np.cos(rlon2 - rlon1)

        return np.arctan2(y, x)
