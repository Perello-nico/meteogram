# meteogram

`meteogram` builds interactive Plotly meteograms from numerical weather model
fields. It can be used as a Python plotting helper, or as a CLI pipeline that:

1. reads a YAML configuration and a CSV list of events;
2. downloads model fields through `drops2`;
3. selects the nearest grid point, or points within a radius, for each event;
4. computes optional derived variables;
5. writes one HTML and one JSON Plotly meteogram per event.

The current implementation is focused on one model at a time and supports
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

The plotting layer can be used without running the collection pipeline:

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

- `MeteogramBuilder`, `PanelSpec`, `SeriesSpec`, `TimeBandSpec`
- `plot_meteogram`
- `collect_data`
- `EventsCollection`, `select_data`

## Notes

- The collector retries `drops2` calls up to `MAX_ITER_DROPS` times.
- Event times are normalized to UTC.
- If a CSV event geometry uses a CRS other than EPSG:4326, pass the source
  EPSG when using `EventsCollection.from_csv` from Python.
- With the default selector radius of `-1`, the nearest model grid point is
  selected for each event.
