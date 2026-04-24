# %%
from __future__ import annotations

import os
import sys
import yaml
from datetime import datetime
from typing import Optional
from meteogram.utils import setup_logger, open_drops_door
from meteogram.collector import Model
from meteogram import collect_data, EventsCollection, select_data, plot_meteogram

# %%
def main(
    path_config: str,
    path_events: str,
    time_from: Optional[datetime | str] = None,
    time_to: Optional[datetime | str] = None,
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
    save_temporary_data = config.get('save_temporary_data', False)

    # logging initialization
    save_log = bool(config.get('log', False))
    log_level = config.get('log_level', 'error')
    log_name = config.get('log_name', 'meteogram')
    log_path = None
    if save_log:
        log_path = os.path.join(path_save_abs, f"{log_name}.log")
    logger = setup_logger(log_path, log_level)

    logger.info('config path: %s', path_config_abs)
    logger.info('output path: %s', path_save_abs)

    # get models
    model_cfg_list = config.get('model', [])
    if len(model_cfg_list) != 1:
        logger.error('Too many models provided - currently only one model is supported')
    model_cfg = model_cfg_list[0]

    # create model
    model = Model.from_dict(model_cfg)

    # open drops door
    url = config.get('dds_url')
    user = config.get('dds_user')
    password = config.get('dds_password')
    if url and user and password:
        open_drops_door(url, user, password)
    else:
        logger.warning('skipping DDS authentication')

    # generate events
    events = EventsCollection.from_csv(path_events, time_from, time_to)

    time_from_sel, time_to_sel = events.timebox()

    data, metadata = collect_data(
        model=model,
        time_from=time_from_sel,
        time_to=time_to_sel,
        logger=logger,
    )

    # select data in the event
    data_selection = select_data(data, events)

    # save data
    if save_temporary_data:
        data.to_netcdf(os.path.join(path_save_abs, "data_model.nc"))    
        metadata.to_csv(os.path.join(path_save_abs, "metadata.csv"), index=False)
        data_selection.to_csv(os.path.join(path_save_abs, "data_selection.csv"), index=False)

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
    path_events = sys.argv[2]
    time_from = sys.argv[3]
    time_to = sys.argv[4]
    main(path_config, path_events, time_from, time_to)
