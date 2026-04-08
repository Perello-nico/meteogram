# %%
from __future__ import annotations

from datetime import datetime, timedelta
from math import cos, sin
import numpy as np
import xarray as xr
from meteogram import MeteogramBuilder, PanelSpec, SeriesSpec

# %%
def main() -> None:

    data = xr.open_dataset("test/test_data.nc")

    times = data.time.values

    # lat/lon selection
    lat_sel = 39.0
    lon_sel = 16.0

    lats = data.latitude.values
    lons = data.longitude.values

    # lats and lons are 2D, I want row, col indices of the point closest to lat_sel/lon_sel
    lat_diff = lats - lat_sel
    lon_diff = lons - lon_sel
    distance = (lat_diff**2 + lon_diff**2) ** 0.5
    row, col = np.unravel_index(distance.argmin(), distance.shape)
    selected_lat = float(lats[row, col])
    selected_lon = float(lons[row, col])

    data_sel = data.sel(rows=row, cols=col)
    temperature_k = data_sel['2t'].values
    dew_point_k = data_sel['2d'].values
    temperature = temperature_k - 273.15
    dew_point = dew_point_k - 273.15
    humidity = data_sel['rh'].values
    wind_10u = data_sel['10u'].values
    wind_10v = data_sel['10v'].values
    wind_speed = np.sqrt(wind_10u**2 + wind_10v**2)
    wind_direction = np.arctan2(wind_10v, wind_10u) * 180 / np.pi + 180
    wind_arrow_level = np.full_like(wind_speed, wind_speed.max() * 1.08)


    builder = MeteogramBuilder(
        times,
        title=f"Latitude: {selected_lat:.5f}, Longitude: {selected_lon:.5f}",
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
                        line={
                            "color": "#ff3b30",
                            "width": 2
                        },
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": "%{customdata:.1f} °C",
                            "customdata": temperature,
                        },
                    ),
                    SeriesSpec(
                        "Dew Point (2m)",
                        dew_point,
                        line={
                            "color": "#9b6bd6",
                            "width": 2
                        },
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": "%{customdata:.1f} °C",
                            "customdata": dew_point,
                        },
                    ),
                    SeriesSpec(
                        "Relative Humidity (2m)",
                        humidity,
                        line={
                            "color": "rgba(46, 134, 222, 0.0)",
                            "width": 0
                        },
                        fill="tozeroy",
                        fillcolor="rgba(46, 134, 222, 0.20)",
                        secondary_y=True,
                        render_order=-1,
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": "%{customdata:.1f}%",
                            "customdata": humidity,
                        },
                    )
                ],
            ),
            PanelSpec(
                title="Wind",
                yaxis_title="m/s",
                yaxis_range=[0, float(wind_speed.max() * 1.18)],
                series=[
                    SeriesSpec(
                        "Wind Speed (10m)",
                        wind_speed,
                        line={
                            "color": "#2ecc71",
                            "width": 2
                        },
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": "%{customdata:.1f} m/s",
                            "customdata": wind_speed,
                        }
                    ),
                    SeriesSpec(
                        "Wind Direction (10m)",
                        wind_arrow_level,
                        mode="markers",
                        marker={
                            "symbol": "arrow",
                            "size": 12,
                            "color": "#2d3436",
                            "line": {"width": 1, "color": "#2d3436"},
                        },
                        marker_angles=wind_direction,
                        showlegend=False,
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": "%{customdata:.0f}°",
                            "customdata": wind_direction,
                        },
                    ),
                ],
            ),
        ]
    )

    figure = builder.to_figure()
    figure.write_html("test/meteogram_demo.html")
    print("Wrote meteogram_demo.html")


if __name__ == "__main__":
    main()
