from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


#####################################################################
# OBJECTS AND UTILS FOR BUILDING A METEOGRAM FIGURE
#####################################################################

@dataclass
class SeriesSpec:
    """One data series rendered inside a meteogram subplot."""

    name: str
    values: Sequence[Any]
    mode: str = "lines"
    line: Optional[Mapping[str, Any]] = None
    marker: Optional[Mapping[str, Any]] = None
    fill: Optional[str] = None
    fillcolor: Optional[str] = None
    stackgroup: Optional[str] = None
    opacity: Optional[float] = None
    marker_angles: Optional[Sequence[float]] = None
    secondary_y: bool = False
    showlegend: bool = True
    render_order: int = 0
    trace_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_trace(self, x: Sequence[Any]) -> Any:
        trace_kwargs = dict(self.trace_kwargs)
        trace_type = trace_kwargs.pop("type", "scatter")
        marker = dict(self.marker or {})
        if self.marker_angles is not None:
            marker["angle"] = list(self.marker_angles)

        if trace_type == "bar":
            return go.Bar(
                x=x,
                y=self.values,
                name=self.name,
                marker=marker or None,
                opacity=self.opacity,
                showlegend=self.showlegend,
                **trace_kwargs,
            )

        return go.Scatter(
            x=x,
            y=self.values,
            name=self.name,
            mode=self.mode,
            line=self.line,
            marker=marker or None,
            fill=self.fill,
            fillcolor=self.fillcolor,
            stackgroup=self.stackgroup,
            opacity=self.opacity,
            showlegend=self.showlegend,
            **trace_kwargs,
        )


@dataclass
class PanelSpec:
    """A subplot definition containing one or more series."""

    title: str
    series: Sequence[SeriesSpec]
    yaxis_title: Optional[str] = None
    height_ratio: float = 1.0
    secondary_y_title: Optional[str] = None
    yaxis_range: Optional[Sequence[float]] = None
    secondary_y_range: Optional[Sequence[float]] = None
    hide_yaxis: bool = False


@dataclass
class TimeBandSpec:
    """Vertical background band applied to selected hours of the day."""

    start_hour: int
    end_hour: int
    fillcolor: str
    opacity: float = 0.12
    rows: Optional[Sequence[int]] = None
    line_width: int = 0


class MeteogramBuilder:
    """Incrementally build a meteogram figure."""

    def __init__(
        self,
        x: Sequence[Any],
        *,
        title: Optional[str] = None,
        template: str = "plotly_white",
        width: int = 1100,
        height_per_panel: int = 220,
        row_spacing: float = 0.08,
        xaxis_title: str = "Time",
        legend: Optional[Mapping[str, Any]] = None,
        time_bands: Optional[Sequence["TimeBandSpec"]] = None,
        now_time: Optional[Any] = None,
    ) -> None:
        self.x = x
        self.title = title
        self.template = template
        self.width = width
        self.height_per_panel = height_per_panel
        self.row_spacing = row_spacing
        self.xaxis_title = xaxis_title
        self.legend = legend
        self.time_bands = time_bands
        self.now_time = now_time
        self._panels = []

    def add_panel(self, panel):
        _validate_panel_lengths(self.x, panel)
        self._panels.append(panel)
        return self

    def extend(self, panels):
        for panel in panels:
            self.add_panel(panel)
        return self

    @property
    def panels(self) -> Tuple[PanelSpec, ...]:
        return tuple(self._panels)

    def to_figure(self):
        return create_meteogram(
            self.x,
            self._panels,
            title=self.title,
            template=self.template,
            width=self.width,
            height_per_panel=self.height_per_panel,
            row_spacing=self.row_spacing,
            xaxis_title=self.xaxis_title,
            legend=self.legend,
            time_bands=self.time_bands,
            now_time=self.now_time,
        )


def _panel_uses_secondary_y(panel: PanelSpec) -> bool:
    return any(series.secondary_y for series in panel.series)


def _validate_panel_lengths(x: Sequence[Any], panel: PanelSpec) -> None:
    expected = len(x)
    for series in panel.series:
        if len(series.values) != expected:
            raise ValueError(
                f"Series '{series.name}' has {len(series.values)} points, expected {expected}."
            )


def _add_panel_to_figure(figure, **kwargs):
    x = kwargs["x"]
    panel = kwargs["panel"]
    row = kwargs["row"]
    for series in sorted(panel.series, key=lambda series: series.render_order):
        figure.add_trace(series.to_trace(x), row=row, col=1, secondary_y=series.secondary_y)

    figure.update_yaxes(title_text=panel.yaxis_title, range=panel.yaxis_range, row=row, col=1)
    if panel.hide_yaxis:
        figure.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title_text=None,
            row=row,
            col=1,
        )

    if _panel_uses_secondary_y(panel):
        figure.update_yaxes(
            title_text=panel.secondary_y_title,
            range=panel.secondary_y_range,
            showgrid=False,
            row=row,
            col=1,
            secondary_y=True,
        )


def create_meteogram(
    x: Sequence[Any],
    panels: Sequence[PanelSpec],
    *,
    title: Optional[str] = None,
    template: str = "plotly_white",
    width: int = 1100,
    height_per_panel: int = 220,
    row_spacing: float = 0.05,
    xaxis_title: str = "Time",
    legend: Optional[Mapping[str, Any]] = None,
    time_bands: Optional[Sequence["TimeBandSpec"]] = None,
    now_time: Optional[Any] = None,
):
    """Compose multiple panel definitions into one meteogram figure."""
    if not panels:
        raise ValueError("At least one panel is required to create a meteogram.")

    for panel in panels:
        _validate_panel_lengths(x, panel)

    figure = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=row_spacing,
        row_heights=[panel.height_ratio for panel in panels],
        specs=[[{"secondary_y": _panel_uses_secondary_y(panel)}] for panel in panels],
        subplot_titles=[panel.title for panel in panels],
    )

    for index, panel in enumerate(panels, start=1):
        _add_panel_to_figure(figure, x=x, panel=panel, row=index)

    _style_figure(
        figure,
        title_text=title,
        template=template,
        panel_count=len(panels),
        xaxis_title=xaxis_title,
        width=width,
        height=height_per_panel * len(panels),
        legend=legend,
        x=x,
    )
    _add_time_bands(figure, x=x, panel_count=len(panels), time_bands=time_bands)
    _add_midnight_lines(figure, x=x, panel_count=len(panels))
    _add_now_line(figure, panel_count=len(panels), now_time=now_time)
    return figure


def _style_figure(
    figure: go.Figure,
    *,
    title_text: Optional[str],
    template: str,
    panel_count: int,
    xaxis_title: str,
    x: Sequence[Any],
    width: int = 1100,
    height: int = 320,
    legend: Optional[Mapping[str, Any]] = None,
) -> None:
    x_values = list(x)
    figure.update_layout(
        template=template,
        title=title_text,
        width=width,
        height=height,
        hovermode="x unified",
        legend=legend or {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 70, "r": 70, "t": 120, "b": 60},
    )
    figure.update_xaxes(
        showgrid=True,
        showline=True,
        mirror=True,
        linecolor="#444444",
        linewidth=1,
        range=[x_values[0], x_values[-1]],
    )
    figure.update_xaxes(title_text=xaxis_title, row=panel_count, col=1)
    figure.update_yaxes(
        showgrid=True,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="#444444",
        linewidth=1,
    )
    _style_secondary_yaxes(figure)


def _add_time_bands(figure, x, panel_count, time_bands):
    if not time_bands:
        return

    datetimes = [_coerce_datetime(value) for value in x]
    if not datetimes:
        return

    edges = _datetime_edges(datetimes)
    for band in time_bands:
        intervals = _band_intervals(datetimes, edges, band)
        if not intervals:
            continue

        rows = list(band.rows) if band.rows is not None else list(range(1, panel_count + 1))
        for row in rows:
            for x0, x1 in intervals:
                figure.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=band.fillcolor,
                    opacity=band.opacity,
                    line_width=band.line_width,
                    layer="below",
                    row=row,
                    col=1,
                )


def _band_intervals(datetimes, edges, band):
    intervals = []
    start_index = None

    for index, moment in enumerate(datetimes):
        if _hour_in_band(moment.hour, band.start_hour, band.end_hour):
            if start_index is None:
                start_index = index
        elif start_index is not None:
            intervals.append((edges[start_index], edges[index]))
            start_index = None

    if start_index is not None:
        intervals.append((edges[start_index], edges[-1]))

    return intervals


def _datetime_edges(datetimes):
    if len(datetimes) == 1:
        return [
            datetimes[0] - timedelta(minutes=30),
            datetimes[0] + timedelta(minutes=30),
        ]

    edges = [datetimes[0] - (datetimes[1] - datetimes[0]) / 2]
    for index in range(len(datetimes) - 1):
        edges.append(datetimes[index] + (datetimes[index + 1] - datetimes[index]) / 2)
    edges.append(datetimes[-1] + (datetimes[-1] - datetimes[-2]) / 2)
    return edges


def _hour_in_band(hour, start_hour, end_hour):
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _coerce_datetime(value):
    if isinstance(value, datetime):
        return value

    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        converted = to_pydatetime()
        if isinstance(converted, datetime):
            return converted

    iso_value = str(value).replace("Z", "+00:00")
    if "T" not in iso_value and " " in iso_value:
        iso_value = iso_value.replace(" ", "T", 1)
    if "." in iso_value:
        prefix, suffix = iso_value.split(".", 1)
        timezone = ""
        if "+" in suffix:
            fraction, timezone = suffix.split("+", 1)
            timezone = "+" + timezone
        elif "-" in suffix:
            fraction, timezone = suffix.split("-", 1)
            timezone = "-" + timezone
        else:
            fraction = suffix
        iso_value = prefix + "." + fraction[:6] + timezone
    return datetime.fromisoformat(iso_value)


def _style_secondary_yaxes(figure):
    for axis_name in figure.layout:
        if not axis_name.startswith("yaxis"):
            continue

        axis = figure.layout[axis_name]
        if getattr(axis, "overlaying", None):
            axis.showgrid = False
            axis.showline = True
            axis.mirror = True
            axis.linecolor = "#444444"
            axis.linewidth = 1


def _add_now_line(figure, panel_count, now_time):
    if now_time is None:
        return

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Now",
            line={"color": "#111111", "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    x_value = _coerce_datetime(now_time)
    for row in range(1, panel_count + 1):
        figure.add_vline(
            x=x_value,
            line_color="#111111",
            line_width=1.5,
            line_dash="dash",
            row=row,
            col=1,
        )


def _add_midnight_lines(figure, x, panel_count):
    datetimes = [_coerce_datetime(value) for value in x]
    midnight_times = []
    for moment in datetimes:
        if (
            moment.hour == 0
            and moment.minute == 0
            and moment.second == 0
            and moment.microsecond == 0
        ):
            midnight_times.append(moment)

    for x_value in midnight_times:
        for row in range(1, panel_count + 1):
            figure.add_vline(
                x=x_value,
                line_color="#111111",
                opacity=0.2,
                line_dash="dot",
                line_width=1.8,
                row=row,
                col=1,
            )


#####################################################################
# PLOT SETTINGS
#####################################################################

WIDTH_LINE = 2

COLOR_T = "#ff3b30"
COLOR_TD = "#9b6bd6"
COLOR_RH = "rgba(46, 134, 222, 0.30)"
COLOR_WS = "#2ecc71"
COLOR_WD = "#2d3436"

UM_T = "°C"
UM_RH = "%"
UM_WS = "m/s"
UM_WD = "°"

NIGHT_TIME = [18, 6]
COLOR_NIGHT = "#95a0a4"

DAY_TIME = [6, 18]
COLOR_DAY = "#ffffff"

#####################################################################
# PLOT FUNCTIONS
#####################################################################


def t_rh_panel(
        temperature: Sequence[Any],
        humidity: Sequence[Any],
        dew_point: Optional[Sequence[Any]] = None,
        color_t: str = COLOR_T,
        color_td: str = COLOR_TD,
        color_rh: str = COLOR_RH,
        um_t: str = UM_T,
        um_rh: str = UM_RH,
) -> PanelSpec:
    """Definition of the temperature and relative humidity panel."""

    t_max = max(temperature)*1.18
    t_min = min(temperature)*0.82
    if dew_point is not None:
        td_max = max(dew_point)*1.18
        td_min = min(dew_point)*0.82
    else:
        td_max = t_max
        td_min = t_min
    y_t_max = max([t_max, td_max])
    y_t_min = min([t_min, td_min])

    series = []

    rh_series = SeriesSpec(
                    "Relative Humidity",
                    humidity,
                    line={"color": color_rh, "width": WIDTH_LINE},
                    opacity=0.0,  # invisible line
                    fill="tozeroy",
                    fillcolor=color_rh,
                    secondary_y=True,
                    render_order=-1,  # BACKGROUND
                    trace_kwargs={
                        # information for the hover tooltip
                        "hovertemplate": f"%{{customdata:.1f}} {um_rh}",
                        "customdata": humidity,
                    },
                )
    series.append(rh_series)

    t_series = SeriesSpec(
                    "Temperature",
                    temperature,
                    line={"color": color_t, "width": WIDTH_LINE},
                    trace_kwargs={
                        # information for the hover tooltip
                        "hovertemplate": f"%{{customdata:.1f}} {um_t}",
                        "customdata": temperature,
                    }
                )
    series.append(t_series)

    if dew_point is not None:
        td_series = SeriesSpec(
                        "Dew Point",
                        dew_point,
                        line={"color": color_td, "width": WIDTH_LINE},
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": f"%{{customdata:.1f}} {um_t}",
                            "customdata": dew_point,
                        },
                    )
        series.append(td_series)

    return PanelSpec(
                title="Temperature and relative humidity",
                yaxis_title=um_t,
                yaxis_range=[y_t_min, y_t_max],
                secondary_y_title=um_rh,
                secondary_y_range=[0, 100],
                series=series,
            )


def wind_panel(
        wind_speed: Sequence[Any],
        wind_direction: Optional[Sequence[Any]] = None,
        color_ws: str = COLOR_WS,
        color_wd: str = COLOR_WD,
        um_ws: str = UM_WS,
        um_wd: str = UM_WD
) -> PanelSpec: 
    """Definition of wind panel"""
    series = []   
    ws_series = SeriesSpec(
                    "Wind Speed",
                    wind_speed,
                    line={"color": color_ws, "width": WIDTH_LINE},
                    trace_kwargs={
                        # information for the hover tooltip
                        "hovertemplate": f"%{{customdata:.1f}} {um_ws}",
                        "customdata": wind_speed,
                    }
                )
    series.append(ws_series)
    if wind_direction is not None:
        wind_arrow_level = np.full_like(wind_speed, max(wind_speed) * 1.08)
        wd_series = SeriesSpec(
                        "Wind Direction",
                        wind_arrow_level,  # type: ignore
                        mode="markers",
                        marker={
                            "symbol": "arrow",
                            "size": 12,
                            "color": color_wd,
                            "line": {"width": 1, "color": color_wd},
                        },
                        marker_angles=wind_direction,
                        showlegend=True,
                        trace_kwargs={
                            # information for the hover tooltip
                            "hovertemplate": f"%{{customdata:.0f}} {um_wd}",
                            "customdata": wind_direction,
                        },
                    )
        series.append(wd_series)
    return PanelSpec(
                title="Wind",
                yaxis_title=um_ws,
                yaxis_range=[0, max(wind_speed) * 1.18],
                series=series
            )

def plot_meteogram(
        times,
        temperature,
        dew_point,
        humidity,
        wind_speed,
        wind_direction,
        time_now=None,
        title='Meteogram',
        nighttime=NIGHT_TIME,
        daytime=DAY_TIME,
        night_color=COLOR_NIGHT,
        day_color=COLOR_DAY
):
    
    # time bands
    night_band = TimeBandSpec(
                    start_hour=nighttime[0],
                    end_hour=nighttime[1],
                    fillcolor=night_color,
                    opacity=0.10,
                )
    day_band = TimeBandSpec(
                    start_hour=daytime[0],
                    end_hour=daytime[1],
                    fillcolor=day_color,
                    opacity=0.10,
                )
    time_bands = [night_band, day_band]

    # generate frame
    builder = MeteogramBuilder(
        times,
        title=title,
        xaxis_title="Time (UTC)",
        height_per_panel=230,
        row_spacing=0.1,
        now_time=time_now,
        time_bands=time_bands,
    )
    
    # generate panels
    panel_t_rh = t_rh_panel(
        temperature=temperature,
        dew_point=dew_point,
        humidity=humidity,
    )

    panel_wind = wind_panel(
        wind_speed=wind_speed,
        wind_direction=wind_direction
    )

    builder.extend(
        [
            panel_t_rh,
            panel_wind
        ]
    )

    figure = builder.to_figure()

    return figure
