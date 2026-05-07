# %%
from __future__ import annotations

import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from meteogram import (
    DerivateInputSpec,
    DerivateSpec,
    EventsCollection,
    Model,
    PanelVariableSpec,
    PlotPanelSpec,
    open_drops_door,
    run_meteogram_workflow,
    setup_logger,
)
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
def _derivate_specs_from_config(config: list[dict]) -> list[DerivateSpec]:
    specs = []
    for derivate in config:
        specs.append(
            DerivateSpec(
                id=derivate["id"],
                function=derivate["function"],
                from_variables=[
                    DerivateInputSpec(id=variable["id"], label=variable["label"])
                    for variable in derivate.get("from", [])
                ],
            )
        )
    return specs


def _plot_panel_specs_from_config(config: list[dict]) -> list[PlotPanelSpec]:
    specs = []
    for panel in config:
        specs.append(
            PlotPanelSpec(
                type=panel["type"],
                variables=[
                    PanelVariableSpec(id=variable["id"], label=variable["label"])
                    for variable in panel.get("variables", [])
                ],
            )
        )
    return specs


def _model_output_name(base: str, model_id: str, one_model: bool, extension: str) -> str:
    if one_model:
        return f"{base}.{extension}"
    return f"{base}_{model_id}.{extension}"


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
    if len(model_cfg_list) == 0:
        logger.error('At least one model must be provided')
        return
    models = [Model.from_dict(model_cfg) for model_cfg in model_cfg_list]

    # generate events
    events = EventsCollection.from_csv(path_events, time_from, time_to)  # type: ignore

    result = run_meteogram_workflow(
        models=models,
        events=events,
        derivates=_derivate_specs_from_config(config.get('derivates', [])),
        plot_panels=_plot_panel_specs_from_config(plot_panels),
        radius_km=config.get('radius_km', -1),
        logger=logger,
    )

    one_model = len(models) == 1

    for model in models:
        metadata = result.metadata[model.id]
        metadata.to_json(
            os.path.join(
                path_save_abs,
                _model_output_name("metadata", model.id, one_model, "json"),
            ),
            date_format='iso',
        )

        if save_temporary_data:
            result.data[model.id].to_netcdf(
                os.path.join(
                    path_save_abs,
                    _model_output_name("data_model", model.id, one_model, "nc"),
                )
            )
            result.selections[model.id].to_json(
                os.path.join(
                    path_save_abs,
                    _model_output_name("data_selection", model.id, one_model, "json"),
                ),
                date_format='iso',
            )

        for event_id, figure in result.figures.get(model.id, {}).items():
            name = f"meteogram_{event_id}" if one_model else f"meteogram_{model.id}_{event_id}"
            figure.write_html(os.path.join(path_save_abs, f"{name}.html"))
            figure.write_json(os.path.join(path_save_abs, f"{name}.json"))


if __name__ == "__main__":
    main()

# %%
