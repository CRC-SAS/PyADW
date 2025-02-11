
import sys
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio
import matplotlib
import matplotlib.pyplot as plt

from types import SimpleNamespace

from src.ADW_numba import ADW


logging.basicConfig(
    format='%(asctime)s - %(name)-5s - %(levelname)-4s -- %(message)s',
    datefmt='%m-%d %H:%M:%S')


EXTENT = SimpleNamespace()
setattr(EXTENT, 'xmin', -80.0)
setattr(EXTENT, 'xmax', -33.5)
setattr(EXTENT, 'ymin', -57.0)
setattr(EXTENT, 'ymax', -7.5)

ROLLING_WINDOW_FILLNA = 21
ROLLING_WINDOW_SMOOTHING = 5

EXAMPLE_DATA = './data/example-data.csv'
EXAMPLE_SHAPE = './data/example-shapefile.shp'

GEN_COMPARATIVE_PLOTS = False


def graficar_raster(geo_df: gpd.GeoDataFrame, raster: xr.Dataset, filename: str,
                    show_plot: bool = True, clip_raster: bool = True) -> None:
    # crear figura
    fig, ax = plt.subplots()
    # agregar elementos a la figura
    geo_df.plot(ax=ax, color='gray', edgecolor='black', linewidth=.5)
    raster.plot(ax=ax) if not clip_raster else raster.rio.clip(geo_df.geometry.values, geo_df.crs).plot(ax=ax)
    # guardar figura
    plt.savefig(f'./data/{filename}')
    # mostrar figura
    plt.show() if show_plot else plt.close()


def main() -> int:
    """Doc"""

    # configurar matplotlib
    matplotlib.use('TkAgg')

    # definir logger
    logger = logging.getLogger('PyADW')

    # definir nivel del logger
    logger.setLevel(logging.DEBUG)

    # leer datos de ejemplo
    logger.info('Leer datos de ejemplo')
    prcp_acumulada_df =  pd.read_csv(EXAMPLE_DATA)
    crcsas_gdf = gpd.read_file(EXAMPLE_SHAPE)

    # convertir prcp_acumulada a geojson
    logger.info('Inicia creación del geodataframe')
    prcp_acumulada_gdf = gpd.GeoDataFrame(
        data=prcp_acumulada_df,
        geometry=gpd.points_from_xy(prcp_acumulada_df.lon_dec, prcp_acumulada_df.lat_dec),
        crs='EPSG:4326')

    # remover filas con NaN en la columna valor, para acelerar interpolación
    prcp_acumulada_gdf = prcp_acumulada_gdf.dropna()

    # interpolar la precipitación acumulada
    logger.info('Preparando interpolación')
    adw = ADW(
        obslon=prcp_acumulada_gdf.lon_dec.to_numpy(dtype=np.float64),
        obslat=prcp_acumulada_gdf.lat_dec.to_numpy(dtype=np.float64),
        obs=prcp_acumulada_gdf.valor.to_numpy(dtype=np.float64),
        extent=[EXTENT.xmin, EXTENT.xmax, EXTENT.ymax, EXTENT.ymin],
        res=0.25)
    logger.info('Inicia interpolación')
    adw.interpolate()
    logger.info('Finaliza interpolación')

    # definir transform
    logger.info('Inicia definición de la transformación a ser utilizada')
    transform = rasterio.transform.from_bounds(
        west=EXTENT.xmin, south=EXTENT.ymin, east=EXTENT.xmax, north=EXTENT.ymax,
        width=adw.grid.shape[1], height=adw.grid.shape[0])

    # crear objeto xarray para manejar más fácilmente el raster interpolado
    logger.info('Inicia creación del raster (rioxarray)')
    coords = dict(latitude=adw.yrange, longitude=adw.xrange)
    raster = xr.DataArray(data=adw.grid, coords=coords)\
        .astype('float32')\
        .rio.write_transform(transform)\
        .rio.write_crs('epsg:4326')

    # recortar raster y graficar (raster original)
    graficar_raster(geo_df=crcsas_gdf, raster=raster, filename='example-interp-original.png')

    # generar figuras para comparar diferentes valores de ROLLING_WINDOW_FILLNA
    if GEN_COMPARATIVE_PLOTS:
        for i in [3,9,15,21]:
            # se rellenan los huecos
            raster_1 = raster.fillna(
                raster.rolling(latitude=i, longitude=i, min_periods=1, center=True).mean())
            # recortar raster y graficar (raster sin faltantes)
            graficar_raster(
                geo_df=crcsas_gdf, raster=raster_1, filename=f'example-interp-sin-faltantes-{i}.png', show_plot=False)

    # rellenar posibles huecos
    logger.info('Inicia rellenado de faltantes (huecos en el raster)')
    raster = raster.fillna(
        raster.rolling(latitude=ROLLING_WINDOW_FILLNA,
                       longitude=ROLLING_WINDOW_FILLNA,
                       min_periods=1, center=True).mean())

    # recortar raster y graficar (raster sin faltantes)
    graficar_raster(geo_df=crcsas_gdf, raster=raster, filename='example-interp-sin-faltantes.png')

    # generar figuras para comparar diferentes valores de ROLLING_WINDOW_SMOOTHING
    if GEN_COMPARATIVE_PLOTS:
        for i in [3,5,7,9]:
            # se suaviza el raster
            raster_1 = raster.rolling(latitude=i, longitude=i, min_periods=1, center=True).mean()
            # recortar raster y graficar (raster suavizado)
            graficar_raster(
                geo_df=crcsas_gdf, raster=raster_1, filename=f'example-interp-suavizado-{i}.png', show_plot=False)

    # suavizar raster
    logger.info('Inicia suavizado del raster interpolado')
    raster = raster.rolling(latitude=ROLLING_WINDOW_SMOOTHING,
                            longitude=ROLLING_WINDOW_SMOOTHING,
                            min_periods=1, center=True).mean()

    # recortar raster y graficar (raster suavizado)
    graficar_raster(geo_df=crcsas_gdf, raster=raster, filename='example-interp-suavizado.png')

    return 0


if __name__ == '__main__':
    sys.exit(main())
