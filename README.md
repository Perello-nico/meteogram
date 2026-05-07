# meteogram

`meteogram` builds interactive Plotly meteograms from numerical weather model
fields. It can be used as a Python plotting helper, or as a CLI pipeline that:

1. reads a YAML configuration and a CSV list of events;
2. downloads model fields through `drops2`;
3. selects the nearest grid point, or points within a radius, for each event;
4. computes optional derived variables;
5. writes one HTML and one JSON Plotly meteogram per event.

The collector and workflow API support one or more models and currently include
temperature, relative humidity, combined temperature/humidity, and wind panels.

## Project Layout

```text
.
|-- main.py                  # CLI entry point
|-- meteogram/
|   |-- collector.py         # model/variable definitions and drops2 collection
|   |-- plot.py              # Plotly panel specs and meteogram builder
|   |-- selector.py          # event loading and point selection
|   |-- settings.py          # plotting and download defaults
|   `-- utils.py             # logging, drops2 credentials, derived variables
|-- test/
|   |-- configuration.yaml   # example run configuration
|   `-- events.csv           # example event locations
|-- pyproject.toml
`-- uv.lock
```

## Requirements

- Python `>=3.14`
- Plotly, NumPy, pandas, xarray, scipy, PyYAML, Pydantic,
  pydantic-settings, pytz, and utm
- Runtime access to `drops2` for data discovery/download
- Geospatial runtime packages used by the selector: `pyproj` and `shapely`

The repository has a `uv.lock` and a local `.python-version` set to Python
`3.14.3`.

## Installation

From the repository root:

```bash
uv sync
```

or, with a standard Python environment:

```bash
pip install -e .
```

If your environment does not already provide `drops2`, `pyproj`, and `shapely`,
install them from the package source used by your organization before running
the CLI.

## CLI Usage

After installation, the project exposes the `meteogram` command:

```bash
meteogram --help
```

Typical run:

```bash
meteogram \
  --config test/configuration.yaml \
  --events test/events.csv \
  --time_from 202604231800 \
  --time_to 202604261800
```

`time_from` and `time_to` are UTC timestamps parsed with the format
`YYYYMMDDHHMM`. If they are not supplied, the CLI uses the current time as the
start and a default 72-hour window as the end.

### Events CSV

For CLI runs, the event file must contain at least:

```csv
id,latitude,longitude
id1,39.541,16.587
id2,38.169,15.847
```

The CLI applies the global `--time_from` and `--time_to` window to all events.
When using `EventsCollection.from_csv` directly from Python, per-event
`time_from` and `time_to` columns can also be used.

### Configuration YAML

The configuration controls the output path, logging, model variables, derived
variables, and plot panels. A shortened example:

```yaml
path_save: output
save_temporary_data: true
log: true
log_level: info
radius_km: -1

# Optional, if the DDS endpoint requires authentication.
# dds_url: ...
# dds_user: ...
# dds_password: ...

model:
  - id: ICON_LAMI
    forecast_hours: 72
    analysis_hours: 0
    variables:
      - id: 2t
        level: 2.0
      - id: rh
        level: 2.0
      - id: 10u
        level: 10.0
      - id: 10v
        level: 10.0

derivates:
  - id: 2t_C
    function: from_K_to_C
    from:
      - id: 2t
        label: t
  - id: wind_speed
    function: compute_wind_speed
    from:
      - id: 10u
        label: u
      - id: 10v
        label: v

plot_panels:
  - type: panel_t_rh
    variables:
      - id: 2t_C
        label: temperature
      - id: rh
        label: humidity
  - type: panel_wind
    variables:
      - id: wind_speed
        label: wind_speed
```

`radius_km` controls event/grid selection. Use `-1` for the nearest grid point,
or a positive distance in kilometers to select all grid points within that
radius.

Supported derived functions:

- `from_K_to_C`
- `compute_wind_speed`
- `compute_wind_direction`

Supported panel types:

- `panel_t`
- `panel_rh`
- `panel_t_rh`
- `panel_wind`

The labels under each panel are the argument names expected by the matching
panel function. For example, `panel_t_rh` expects `temperature`, `humidity`,
and optionally `dew_point`.

## Outputs

For each event id, the CLI writes:

- `meteogram_<id>.html`: interactive Plotly figure
- `meteogram_<id>.json`: serialized Plotly figure

It also writes:

- `metadata.json`: selected model reference times
- `meteogram.log`: when `log: true`
- `data_model.nc` and `data_selection.json`: when `save_temporary_data: true`

Relative `path_save` values are resolved from the directory containing the
configuration file.

## Python API

The collector can be used directly with Python objects. A `Model` is the
programmatic equivalent of one item in the YAML `model` list:

```python
from datetime import datetime

from meteogram import Model, Variable, collect_data, open_drops_door, setup_logger

logger = setup_logger(verbosity="info")

# Optional, if the DDS endpoint requires authentication.
open_drops_door("https://dds.example.org", "user", "password")

model = Model(
    id="ICON_LAMI",
    forecast_hours=72,
    analysis_hours=0,
    runs=None,  # Optional list of reference run hours, for example [0, 12].
    variables=[
        Variable(id="2t", level=2.0, description="Temperature at 2m"),
        Variable(id="rh", level=2.0, description="Relative humidity at 2m"),
        Variable(id="10u", level=10.0, description="U wind at 10m"),
        Variable(id="10v", level=10.0, description="V wind at 10m"),
    ],
)

data, metadata = collect_data(
    model=model,
    time_from=datetime(2026, 4, 23, 18),
    time_to=datetime(2026, 4, 26, 18),
    all_variables=True,
    only_last_run=True,
    logger=logger,
)
```

Collector fields:

- `Model.id`: DDS/drops2 coverage id, passed as `data_id`.
- `Model.forecast_hours`: how far before `time_from` to search reference runs.
- `Model.analysis_hours`: how far after `time_to` to search reference runs.
- `Model.runs`: optional list of allowed reference run hours. `None` accepts all runs.
- `Variable.id`: DDS/drops2 variable id, passed as `variable`.
- `Variable.level`: DDS/drops2 level value, passed as `level`.
- `Variable.description`: optional metadata copied to the output data variable.

The full CLI workflow is also available with Python objects and supports
multiple models:

```python
from meteogram import (
    DerivateInputSpec,
    DerivateSpec,
    EventsCollection,
    Model,
    PanelVariableSpec,
    PlotPanelSpec,
    Variable,
    run_meteogram_workflow,
)

models = [
    Model(
        id="ICON_LAMI",
        forecast_hours=72,
        analysis_hours=0,
        variables=[
            Variable(id="2t", level=2.0),
            Variable(id="2d", level=2.0),
            Variable(id="rh", level=2.0),
            Variable(id="10u", level=10.0),
            Variable(id="10v", level=10.0),
        ],
    ),
]

events = EventsCollection.from_csv(
    "events.csv",
    time_from="202604231800",
    time_to="202604261800",
)

result = run_meteogram_workflow(
    models=models,
    events=events,
    radius_km=-1,
    derivates=[
        DerivateSpec(
            id="2t_C",
            function="from_K_to_C",
            from_variables=[DerivateInputSpec(id="2t", label="t")],
        ),
        DerivateSpec(
            id="wind_speed",
            function="compute_wind_speed",
            from_variables=[
                DerivateInputSpec(id="10u", label="u"),
                DerivateInputSpec(id="10v", label="v"),
            ],
        ),
    ],
    plot_panels=[
        PlotPanelSpec(
            type="panel_t_rh",
            variables=[
                PanelVariableSpec(id="2t_C", label="temperature"),
                PanelVariableSpec(id="rh", label="humidity"),
            ],
        ),
        PlotPanelSpec(
            type="panel_wind",
            variables=[PanelVariableSpec(id="wind_speed", label="wind_speed")],
        ),
    ],
)

figure = result.figures["ICON_LAMI"]["id1"]
figure.write_html("meteogram_id1.html")
```

The workflow result contains:

- `data`: collected `xarray.Dataset` objects keyed by model id.
- `metadata`: selected reference times keyed by model id.
- `selections`: event/grid selections as pandas data frames keyed by model id.
- `figures`: Plotly figures keyed by model id and event id.

For lower-level event selection, call `select_data(data, events, radius_km=...)`
directly. Use `radius_km=-1` for the nearest grid point, or a positive radius in
kilometers to keep all grid points within range.

The plotting layer can still be used without running the collection pipeline:

```python
from meteogram import MeteogramBuilder, PanelSpec, SeriesSpec

times = ["2026-04-23 18:00", "2026-04-23 19:00", "2026-04-23 20:00"]

temperature = PanelSpec(
    title="Temperature",
    yaxis_title="deg C",
    series=[
        SeriesSpec("Temperature", [12.4, 11.8, 10.9], line={"color": "#ff3b30"}),
        SeriesSpec("Dew Point", [7.0, 7.3, 7.5], line={"color": "#9b6bd6"}),
    ],
)

humidity = PanelSpec(
    title="Relative Humidity",
    yaxis_title="%",
    series=[
        SeriesSpec("Relative Humidity", [72, 78, 83], line={"color": "#0c53e0"}),
    ],
)

figure = (
    MeteogramBuilder(times, title="Example meteogram")
    .add_panel(temperature)
    .add_panel(humidity)
    .to_figure()
)

figure.write_html("meteogram_example.html")
```

Useful public objects are exported from `meteogram.__init__`:

- `Model`, `Variable`
- `DerivateInputSpec`, `DerivateSpec`, `PanelVariableSpec`, `PlotPanelSpec`
- `MeteogramWorkflowResult`
- `MeteogramBuilder`, `PanelSpec`, `SeriesSpec`, `TimeBandSpec`
- `plot_meteogram`, `get_panel`
- `collect_data`, `collect_models`, `run_meteogram_workflow`
- `EventsCollection`, `select_data`, `select_models_data`
- `apply_derivates`, `build_meteogram_figures`
- `open_drops_door`, `setup_logger`

## Notes

- The collector retries `drops2` calls up to `MAX_ITER_DROPS` times.
- Event times are normalized to UTC.
- If a CSV event geometry uses a CRS other than EPSG:4326, pass the source
  EPSG when using `EventsCollection.from_csv` from Python.
- With the default selector radius of `-1`, the nearest model grid point is
  selected for each event.
