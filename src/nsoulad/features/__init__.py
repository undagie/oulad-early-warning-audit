"""Feature engineering di atas WindowView (Fase 1).

Semua fungsi mengonsumsi WindowView yang sudah bebas-bocor (cutoff-first LEAP) dan
menghasilkan fitur leak-free. Encoder fit-on-train ada di static.StaticEncoder.
"""

from nsoulad.features.assessment import assessment_features
from nsoulad.features.build import (
    WindowDataset,
    build_window_dataset,
    flatten_sequence,
    presentation_split_datasets,
    split_window_datasets,
)
from nsoulad.features.course_structure import course_structure_features
from nsoulad.features.static import StaticEncoder
from nsoulad.features.temporal import (
    SEQ_FEATURES,
    to_sequence_tensor,
    weekly_engagement,
)

__all__ = [
    "weekly_engagement",
    "to_sequence_tensor",
    "SEQ_FEATURES",
    "StaticEncoder",
    "assessment_features",
    "course_structure_features",
    "WindowDataset",
    "build_window_dataset",
    "split_window_datasets",
    "presentation_split_datasets",
    "flatten_sequence",
]
