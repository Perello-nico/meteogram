# %%
from __future__ import annotations

from datetime import datetime, timedelta
from math import cos, sin
import xarray as xr
from meteogram import MeteogramBuilder, PanelSpec, SeriesSpec, create_wind_direction_panel

# %%
def main() -> None:
    start = datetime(2026, 2, 27, 0, 0)
    hours = [start + timedelta(hours=index) for index in range(49)]

    temperature = [10 + 2.5 * sin(index / 4) + 1.2 * cos(index / 2.5) for index in range(49)]
    dew_point = [value - (8 + 1.5 * sin(index / 3)) for index, value in enumerate(temperature)]
    humidity = [42 - 7 * sin(index / 5) - 3 * cos(index / 2.8) for index in range(49)]
    wind_speed = [2.5 + 1.1 * sin(index / 6) - 0.6 * cos(index / 2.2) for index in range(49)]
    wind_gust = [value + 1.4 + 0.8 * sin(index / 4.2) for index, value in enumerate(wind_speed)]
    wind_direction = [(220 + 35 * sin(index / 4.5) - 70 * cos(index / 9.0)) % 360 for index in range(49)]

    builder = MeteogramBuilder(
        hours,
        title="Latitude: XXX, Longitude: YYY",
        height_per_panel=230,
    )
    builder.extend(
        [
            PanelSpec(
                title="Temperature and relative humidity",
                yaxis_title="°C",
                secondary_y_title="%",
                secondary_y_range=[0, 100],
                series=[
                    SeriesSpec(
                        "Temperature (2m)",
                        temperature,
                        line={"color": "#ff3b30", "width": 2}
                    ),
                    SeriesSpec(
                        "Dew Point (2m)",
                        dew_point,
                        line={"color": "#9b6bd6", "width": 2}
                    ),
                    SeriesSpec(
                        "Relative Humidity (2m)",
                        humidity,
                        line={"color": "rgba(46, 134, 222, 0.0)", "width": 0},
                        fill="tozeroy",
                        fillcolor="rgba(46, 134, 222, 0.20)",
                        secondary_y=True,
                        render_order=-1,
                    )
                ],
            ),
            PanelSpec(
                title="Wind",
                yaxis_title="m/s",
                series=[
                    SeriesSpec(
                        "Wind Speed (10m)",
                        wind_speed,
                        line={"color": "#2ecc71", "width": 2}),
                ],
            ),
            create_wind_direction_panel(
                hours,
                wind_direction,
                title="Wind Direction",
                hover_label="Direction (deg)",
            ),
        ]
    )

    figure = builder.to_figure()
    figure.write_html("meteogram_demo.html")
    print("Wrote meteogram_demo.html")


if __name__ == "__main__":
    main()
