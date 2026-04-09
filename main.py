# %%
from __future__ import annotations

from datetime import datetime, timedelta
from math import cos, sin
import numpy as np
import xarray as xr
from meteogram import plot_meteogram

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
    

    time_now = datetime(2026, 2, 28, 18)

    figure = plot_meteogram(
        times,
        temperature,
        dew_point,
        humidity,
        wind_speed,
        wind_direction,
        time_now,
        title='Meteogram'
    )

    figure.write_html("test/meteogram_demo.html")
    print("Wrote meteogram_demo.html")


if __name__ == "__main__":
    main()
