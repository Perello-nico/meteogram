from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
import xarray as xr

from .collector import Model, collect_data
from .plot import get_panel, plot_meteogram
from .selector import EventsCollection, select_data
from .settings import LOGGER
from .utils import get_derivates_fn


@dataclass(frozen=True)
class DerivateInputSpec:
    id: str
    label: str


@dataclass(frozen=True)
class DerivateSpec:
    id: str
    function: str
    from_variables: Sequence[DerivateInputSpec]


@dataclass(frozen=True)
class PanelVariableSpec:
    id: str
    label: str


@dataclass(frozen=True)
class PlotPanelSpec:
    type: str
    variables: Sequence[PanelVariableSpec]


@dataclass
class MeteogramWorkflowResult:
    data: dict[str, xr.Dataset] = field(default_factory=dict)
    metadata: dict[str, pd.DataFrame] = field(default_factory=dict)
    selections: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, dict[int | str, Any]] = field(default_factory=dict)


def collect_models(
    models: Sequence[Model],
    time_from: datetime | str,
    time_to: datetime | str,
    *,
    round_time: bool = True,
    round_unit: str = "hour",
    all_variables: bool = True,
    only_last_run: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, xr.Dataset], dict[str, pd.DataFrame]]:
    """Collect data for multiple models keyed by model id."""
    logger = logger or LOGGER
    _validate_unique_model_ids(models)

    data_by_model: dict[str, xr.Dataset] = {}
    metadata_by_model: dict[str, pd.DataFrame] = {}
    for model in models:
        data, metadata = collect_data(
            model=model,
            time_from=time_from,
            time_to=time_to,
            round_time=round_time,
            round_unit=round_unit,
            all_variables=all_variables,
            only_last_run=only_last_run,
            logger=logger,
        )
        data_by_model[model.id] = data
        metadata_by_model[model.id] = metadata
    return data_by_model, metadata_by_model


def select_models_data(
    data_by_model: dict[str, xr.Dataset],
    events: EventsCollection,
    *,
    radius_km: float = -1,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Select event data for each model dataset."""
    logger = logger or LOGGER
    selections: dict[str, pd.DataFrame] = {}
    for model_id, dataset in data_by_model.items():
        if not dataset.data_vars:
            logger.warning("Skipping selection for %s: empty dataset", model_id)
            selections[model_id] = pd.DataFrame()
            continue
        selections[model_id] = select_data(dataset, events, radius_km=radius_km, logger=logger)
    return selections


def apply_derivates(
    data_selection: pd.DataFrame,
    derivates: Sequence[DerivateSpec],
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return a copy of selected data with configured derived variables added."""
    logger = logger or LOGGER
    result = data_selection.copy()

    for derivate in derivates:
        func_deriv = get_derivates_fn(derivate.function, logger)
        if func_deriv is None:
            continue
        vars_input = {
            variable.label: result[variable.id].values
            for variable in derivate.from_variables
        }
        result[derivate.id] = func_deriv(**vars_input)
    return result


def build_meteogram_figures(
    data_selection: pd.DataFrame,
    events: EventsCollection,
    plot_panels: Sequence[PlotPanelSpec],
    *,
    title_prefix: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[int | str, Any]:
    """Build one Plotly meteogram figure for each event."""
    logger = logger or LOGGER
    figures: dict[int | str, Any] = {}

    for event_id in events.id_list:
        logger.info("Plotting meteogram for event id: %s", event_id)
        data_sel = data_selection[data_selection[events.key_id] == event_id]
        times = data_sel["time"]
        panel_specs = []

        for panel in plot_panels:
            panel_fn = get_panel(panel.type, logger)
            if panel_fn is None:
                continue
            vars_plot = {
                variable.label: data_sel[variable.id].values
                for variable in panel.variables
            }
            panel_specs.append(panel_fn(**vars_plot))

        title = str(event_id) if title_prefix is None else f"{title_prefix} {event_id}"
        figures[event_id] = plot_meteogram(panel_specs, times, title=title)
    return figures


def run_meteogram_workflow(
    models: Sequence[Model],
    events: EventsCollection,
    *,
    derivates: Sequence[DerivateSpec] | None = None,
    plot_panels: Sequence[PlotPanelSpec] | None = None,
    radius_km: float = -1,
    round_time: bool = True,
    round_unit: str = "hour",
    all_variables: bool = True,
    only_last_run: bool = True,
    logger: logging.Logger | None = None,
) -> MeteogramWorkflowResult:
    """Run the programmatic equivalent of the CLI workflow for one or more models."""
    logger = logger or LOGGER
    time_from, time_to = events.timebox()
    data_by_model, metadata_by_model = collect_models(
        models,
        time_from,
        time_to,
        round_time=round_time,
        round_unit=round_unit,
        all_variables=all_variables,
        only_last_run=only_last_run,
        logger=logger,
    )
    selections = select_models_data(
        data_by_model,
        events,
        radius_km=radius_km,
        logger=logger,
    )

    derivates = derivates or []
    for model_id, data_selection in selections.items():
        if data_selection.empty:
            continue
        selections[model_id] = apply_derivates(
            data_selection,
            derivates,
            logger=logger,
        )

    figures: dict[str, dict[int | str, Any]] = {}
    if plot_panels:
        for model_id, data_selection in selections.items():
            if data_selection.empty:
                figures[model_id] = {}
                continue
            figures[model_id] = build_meteogram_figures(
                data_selection,
                events,
                plot_panels,
                title_prefix=model_id,
                logger=logger,
            )

    return MeteogramWorkflowResult(
        data=data_by_model,
        metadata=metadata_by_model,
        selections=selections,
        figures=figures,
    )


def _validate_unique_model_ids(models: Sequence[Model]) -> None:
    model_ids = [model.id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Model ids must be unique when collecting multiple models.")
