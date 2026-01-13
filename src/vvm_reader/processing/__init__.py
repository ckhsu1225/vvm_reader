"""
VVM Reader Data Processing

This package provides data processing functionality including terrain masking,
wind centering, and vertical level processing.
"""

# Terrain processing functions
from .terrain import (
    apply_terrain_mask,
    center_staggered_variables,
)

# Vertical level processing functions
from .vertical import (
    read_vertical_levels_from_fort98,
    read_reference_profiles_from_fort98,
    resolve_vertical_slice,
    resolve_vertical_selection,
    extend_vertical_slice_for_centering,
    expand_indices_for_centering,
    filter_to_final_indices,
    extract_surface_nearest_values,
    validate_vertical_selection,
    apply_vertical_selection,
    ensure_vertical_coordinate_in_meters,
    crop_vertical_after_centering,
    VerticalSelectionInfo,
)

__all__ = [
    # Terrain processing
    "apply_terrain_mask",
    "center_staggered_variables",
    # Vertical level processing
    "read_vertical_levels_from_fort98",
    "read_reference_profiles_from_fort98",
    "resolve_vertical_slice",
    "resolve_vertical_selection",
    "extend_vertical_slice_for_centering",
    "expand_indices_for_centering",
    "filter_to_final_indices",
    "extract_surface_nearest_values",
    "validate_vertical_selection",
    "apply_vertical_selection",
    "ensure_vertical_coordinate_in_meters",
    "crop_vertical_after_centering",
    "VerticalSelectionInfo",
]
