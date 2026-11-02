"""Pipeline data OULAD: schema, loader, labels, windowing (LEAP), splits."""

from nsoulad.data.labels import make_binary_label
from nsoulad.data.loader import OuladTables, load_oulad
from nsoulad.data.windowing import (
    WindowView,
    build_window_view,
    cohort_at_window,
    day_to_week,
    truncate_by_cutoff,
    week_cutoff_day,
)

__all__ = [
    "OuladTables",
    "load_oulad",
    "make_binary_label",
    "WindowView",
    "build_window_view",
    "cohort_at_window",
    "day_to_week",
    "truncate_by_cutoff",
    "week_cutoff_day",
]
