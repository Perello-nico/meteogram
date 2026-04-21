# %%
from __future__ import annotations

import os
import sys
import yaml
from datetime import datetime, timedelta
from math import cos, sin
import numpy as np
import xarray as xr
from meteogram.utils import setup_logger, open_drops_door
from meteogram import plot_meteogram, collect_data

# %%
def main(
    path_config: str,
) -> None:

    # load configuration file
    path_config_abs = os.path.abspath(path_config)
    with open(path_config_abs, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError('configuration file is empty')
    if not isinstance(config, dict):
        raise ValueError('configuration file must contain a YAML mapping')
    path_config_dir = os.path.dirname(path_config_abs)

    # path save data
    path_save = config.get('path_save')
    if path_save is None:
        path_save = '.'  # to save in the configuration file directory
    if os.path.isabs(path_save):
        path_save_abs = path_save
    else:
        path_save_abs = os.path.abspath(os.path.join(path_config_dir, path_save))
    os.makedirs(path_save_abs, exist_ok=True)

    # logging initialization
    save_log = True
    log_path = None
    log_level = 'info'
    log_name = 'meteogram.log'
    log_config = config.get('log')
    if log_config:
        log_config = log_config[0]
        save_log = bool(log_config.get('save', True))
        log_level = log_config.get('level', 'info')
        if save_log:
            log_path = os.path.join(path_save_abs, log_name)
    logger = setup_logger(log_path, log_level)
    logger.info('Config: %s', path_config_abs)
    logger.info('Output: %s', path_save_abs)
    if save_log:
        logger.info('Log file: %s', log_path)

    # get models
    models_list = config.get('models')

    # open drops door
    dds_config = config.get('dds')
    if dds_config:
        dds_config = dds_config[0]
        url = dds_config.get('url')
        user = dds_config.get('user')
        password = dds_config.get('password')
        if url and user and password:
            open_drops_door(url, user, password)
        else:
            logger.warning('Incomplete DDS credentials provided, skipping DDS authentication')
    else:
        logger.warning('DDS credentials not provided, skipping DDS authentication')

    # time test
    time_from = '202604100000'
    time_to = '202604101200'

    collect_data(
        time_from=time_from,
        time_to=time_to,
        models=models_list,  # type: ignore
        path_save=path_save_abs,
        logger=logger,
    )

    # data = xr.open_dataset("test/test_data.nc")

    # times = data.time.values

    # # lat/lon selection
    # lat_sel = 39.0
    # lon_sel = 16.0

    # lats = data.latitude.values
    # lons = data.longitude.values

    # # lats and lons are 2D, I want row, col indices of the point closest to lat_sel/lon_sel
    # lat_diff = lats - lat_sel
    # lon_diff = lons - lon_sel
    # distance = (lat_diff**2 + lon_diff**2) ** 0.5
    # row, col = np.unravel_index(distance.argmin(), distance.shape)
    # selected_lat = float(lats[row, col])
    # selected_lon = float(lons[row, col])

    # data_sel = data.sel(rows=row, cols=col)
    # temperature_k = data_sel['2t'].values
    # dew_point_k = data_sel['2d'].values
    # temperature = temperature_k - 273.15
    # dew_point = dew_point_k - 273.15
    # humidity = data_sel['rh'].values
    # wind_10u = data_sel['10u'].values
    # wind_10v = data_sel['10v'].values
    # wind_speed = np.sqrt(wind_10u**2 + wind_10v**2)
    # wind_direction = np.arctan2(wind_10v, wind_10u) * 180 / np.pi + 180
    

    # time_now = datetime(2026, 2, 28, 18)

    # figure = plot_meteogram(
    #     times,
    #     temperature,
    #     dew_point,
    #     humidity,
    #     wind_speed,
    #     wind_direction,
    #     title='Meteogram'
    # )

    # figure.write_html("test/meteogram_demo.html")
    # print("Wrote meteogram_demo.html")


if __name__ == "__main__":
    path_config = sys.argv[1]
    main(path_config=path_config)
