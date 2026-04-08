# meteogram

Small Plotly helpers for building flexible meteograms from reusable subplot specs.

## Install

```bash
pip install .
```

## Usage

```python
from meteogram import MeteogramBuilder, PanelSpec, SeriesSpec

x = ["2026-02-27 00:00", "2026-02-27 03:00", "2026-02-27 06:00"]

temperature_panel = PanelSpec(
    title="Temperature",
    yaxis_title="°C",
    series=[
        SeriesSpec("Temperature (2m)", [10.2, 11.4, 9.8], line={"color": "#ff3b30"}),
        SeriesSpec("Dew Point (2m)", [1.0, 2.5, 0.8], line={"color": "#9b6bd6"}),
    ],
)

humidity_panel = PanelSpec(
    title="Relative Humidity",
    yaxis_title="%",
    series=[
        SeriesSpec("Relative Humidity", [78, 73, 82], line={"color": "#2e86de"}),
    ],
)

figure = (
    MeteogramBuilder(x, title="Latitude: XXX, Longitude: YYY")
    .add_panel(temperature_panel)
    .add_panel(humidity_panel)
    .to_figure()
)

figure.show()
```

## API

- `SeriesSpec`: one trace inside a subplot.
- `PanelSpec`: one subplot containing one or more series.
- `create_subplot(...)`: render a single subplot as a Plotly figure.
- `create_meteogram(...)`: compose multiple panels into a full meteogram.
- `create_wind_direction_panel(...)`: helper for a compact wind-direction arrow row.
- `MeteogramBuilder`: chainable helper for incrementally building a full figure.
