# %%
from __future__ import annotations

import os
import sys
import yaml
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from meteogram.utils import setup_logger, open_drops_door, get_derivates_fn
from meteogram.collector import Model
from meteogram.plot import get_panel
from meteogram import collect_data, EventsCollection, select_data, plot_meteogram
from meteogram.settings import DEFATUL_DURATION

TIME_RUN = datetime.now()


# %%
class MeteogramCLI(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config: Path = Field(Path('test/configuration.yaml'), description="Path to configuration file (yaml)")
    events: Path = Field(Path('test/events.csv'), description="Path to events file (csv)")
    time_from: datetime = Field(TIME_RUN, description="Time from data selection (UTC)")
    time_to: datetime = Field(TIME_RUN+timedelta(hours=DEFATUL_DURATION), description="Time to data selection (UTC)")

    @field_validator("config", mode="before")
    @classmethod
    def _check_config_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Configuration file not found.")
        return v

    @field_validator("events", mode="before")
    @classmethod
    def _check_events_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Events file not found.")
        return v

    @field_validator("time_from", mode="before")
    @classmethod
    def _transform_time_from(cls, v: str | datetime) -> datetime:
        if isinstance(v, str):
            v = datetime.strptime(v, "%Y%m%d%H%M")
        return v

    @field_validator("time_to", mode="before")
    @classmethod
    def _transform_time_to(cls, v: str | datetime) -> datetime:
        if isinstance(v, str):
            v = datetime.strptime(v, "%Y%m%d%H%M")
        return v

# %%
def main() -> None:

    cli = MeteogramCLI()  # type: ignore

    path_config = cli.config
    path_events = cli.events
    time_from = cli.time_from
    time_to = cli.time_to

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
    path_save = config.get('path_save', '.')  # default: save in the configuration file directory
    if os.path.isabs(path_save):
        path_save_abs = path_save
    else:
        path_save_abs = os.path.abspath(os.path.join(path_config_dir, path_save))
    os.makedirs(path_save_abs, exist_ok=True)

    # by default, do not save temporary data
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

    # check if plot panels are present
    plot_panels = config.get('plot_panels', [])
    if len(plot_panels) == 0:
        logger.error("Plot panels missing in configuration file")
        return

    # open drops door
    url = config.get('dds_url')
    user = config.get('dds_user')
    password = config.get('dds_password')
    if url and user and password:
        logger.info('DDS authentication')
        open_drops_door(url, user, password)
    else:
        logger.warning('skipping DDS authentication')

    # get models
    model_cfg_list = config.get('model', [])
    if len(model_cfg_list) != 1:
        logger.error('Too many models provided - currently only one model is supported - take first one')
    model_cfg = model_cfg_list[0]

    # create model
    model = Model.from_dict(model_cfg)

    # generate events
    events = EventsCollection.from_csv(path_events, time_from, time_to)  # type: ignore

    # time range for selection
    time_from_sel, time_to_sel = events.timebox()

    # data collection from drops
    data, metadata = collect_data(
        model=model,
        time_from=time_from_sel,
        time_to=time_to_sel,
        logger=logger,
    )

    # save metadata
    metadata.to_json(os.path.join(path_save_abs, "metadata.json"), date_format='iso')

    # select data in the event
    data_selection = select_data(data, events)

    # add derivates to the dataset
    derivates = config.get('derivates', [])
    if len(derivates) > 0:
        logger.info('Computing derivates variables')
        for derivate in derivates:
            func_deriv_code = derivate.get('function')
            if func_deriv_code is None:
                logger.error('Function to compute derivate is missing')
                continue
            func_deriv = get_derivates_fn(func_deriv_code, logger)
            vars_input = dict()
            for variable in derivate.get('from'):
                vars_input[variable['label']] = data_selection[variable['id']].values
            output = func_deriv(**vars_input)
            data_selection[derivate['id']] = output

    # save data (optional)
    if save_temporary_data:
        data.to_netcdf(os.path.join(path_save_abs, "data_model.nc"))    
        data_selection.to_json(os.path.join(path_save_abs, "data_selection.json"), date_format='iso')

    # check if all variables needed to plot the meteogram are present
    for panel in plot_panels:
        for var in panel['variables']:
            var_plot = var['id']
            if var_plot not in data_selection.columns:
                logger.error(f"Variable {var_plot} missing - needed for plotting")

    # create meteogram plot for each event
    for id in events.id_list:
        logger.info(f"Plotting meteogram for event id: {id}")

        # data selection
        data_sel = data_selection[data_selection.id == id]
        times = data_sel['time']

        panels_plot_list = []

        for panel in plot_panels:
            panel_code = panel.get('type')
            panel_fn = get_panel(panel_code, logger)
            vars_plot = dict()
            for variable in panel.get('variables'):
                vars_plot[variable['label']] = data_sel[variable['id']].values
            panel_plot = panel_fn(**vars_plot)
            panels_plot_list.append(panel_plot)

        figure = plot_meteogram(
            panels_plot_list,
            times,
            title=f'{id}'
        )

        figure.write_html(os.path.join(path_save_abs, f"meteogram_{id}.html"))
        figure.write_json(os.path.join(path_save_abs, f"meteogram_{id}.json"))


if __name__ == "__main__":
    main()

# %%
