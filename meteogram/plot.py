from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def create_subplot(
    x: Sequence[Any],
    panel: PanelSpec,
    *,
    template: str = "plotly_white",
    xaxis_title: str = "Time",
    time_bands: Optional[Sequence["TimeBandSpec"]] = None,
    now_time: Optional[Any] = None,
):
    """Create a single subplot figure from one panel definition."""
    _validate_panel_lengths(x, panel)

    figure = make_subplots(specs=[[{"secondary_y": _panel_uses_secondary_y(panel)}]])
    _add_panel_to_figure(figure, x=x, panel=panel, row=1)
    _style_figure(
        figure,
        title_text=panel.title,
        template=template,
        panel_count=1,
        xaxis_title=xaxis_title,
        x=x,
    )
    _add_time_bands(figure, x=x, panel_count=1, time_bands=time_bands)
    _add_now_line(figure, panel_count=1, now_time=now_time)
    return figure


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
    _add_now_line(figure, panel_count=len(panels), now_time=now_time)
    return figure


def create_wind_direction_panel(
    x: Sequence[Any],
    directions: Sequence[float],
    *,
    name: str = "Wind Direction (10m)",
    title: str = "Wind Direction",
    height_ratio: float = 0.35,
    marker: Optional[Mapping[str, Any]] = None,
    hover_values: Optional[Sequence[Any]] = None,
    hover_label: str = "Direction",
):
    """Create a compact panel with rotated arrow markers for wind direction."""
    if len(directions) != len(x):
        raise ValueError(
            "Wind direction series must have the same number of points as the x-axis."
        )

    base_marker = {
        "symbol": "arrow",
        "size": 14,
        "color": "#333333",
        "angle": list(directions),
        "line": {"width": 1, "color": "#333333"},
    }
    if marker:
        base_marker.update(marker)

    trace_kwargs = {
        "hovertemplate": "%{x}<br>" + hover_label + ": %{customdata}<extra></extra>",
        "customdata": list(hover_values) if hover_values is not None else list(directions),
    }

    return PanelSpec(
        title=title,
        yaxis_range=[-1, 1],
        height_ratio=height_ratio,
        hide_yaxis=True,
        series=[
            SeriesSpec(
                name=name,
                values=[0] * len(x),
                mode="markers",
                marker=base_marker,
                trace_kwargs=trace_kwargs,
                showlegend=False,
            )
        ],
    )

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
