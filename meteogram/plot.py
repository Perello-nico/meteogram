from dataclasses import dataclass, field
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
    secondary_y: bool = False
    showlegend: bool = True
    render_order: int = 0
    trace_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_trace(self, x: Sequence[Any]) -> Any:
        trace_kwargs = dict(self.trace_kwargs)
        trace_type = trace_kwargs.pop("type", "scatter")

        if trace_type == "bar":
            return go.Bar(
                x=x,
                y=self.values,
                name=self.name,
                marker=self.marker,
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
            marker=self.marker,
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


def create_subplot(
    x: Sequence[Any],
    panel: PanelSpec,
    *,
    template: str = "plotly_white",
    xaxis_title: str = "Time",
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
    )
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
    )
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
        row_spacing: float = 0.05,
        xaxis_title: str = "Time",
        legend: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.x = x
        self.title = title
        self.template = template
        self.width = width
        self.height_per_panel = height_per_panel
        self.row_spacing = row_spacing
        self.xaxis_title = xaxis_title
        self.legend = legend
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
    width: int = 1100,
    height: int = 320,
    legend: Optional[Mapping[str, Any]] = None,
) -> None:
    figure.update_layout(
        template=template,
        title=title_text,
        width=width,
        height=height,
        hovermode="x unified",
        legend=legend or {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 70, "r": 70, "t": 90, "b": 60},
    )
    figure.update_xaxes(showgrid=True, title_text=xaxis_title, row=panel_count, col=1)
    figure.update_yaxes(showgrid=True, zeroline=False)
