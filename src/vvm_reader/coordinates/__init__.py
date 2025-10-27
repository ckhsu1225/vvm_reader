"""
VVM Reader Coordinate Handling

This package provides coordinate system handling functionality including
spatial coordinate processing and time coordinate transformations.
"""

# Spatial coordinate functions
from .spatial import (
    # TOPO loading and extraction
    load_topo_dataset,
    extract_coordinates_from_topo,
    prepare_topo_data,
    extract_terrain_height,
    # Coordinate conversion
    convert_coordinates_to_indices,
    convert_indices_to_coordinates,
    validate_region_bounds,
    # Slicing operations
    compute_regional_slices,
    compute_centering_slices,
    # Dataset operations
    apply_spatial_selection,
    crop_dataset_after_centering,
    assign_spatial_coordinates,
)

# Time coordinate functions
from .time_handler import (
    normalize_time_value,
    normalize_time_range,
    read_time_from_file,
    filter_files_by_time,
)

__all__ = [
    # Spatial - TOPO loading and extraction
    "load_topo_dataset",
    "extract_coordinates_from_topo",
    "prepare_topo_data",
    "extract_terrain_height",
    # Spatial - Coordinate conversion
    "convert_coordinates_to_indices",
    "convert_indices_to_coordinates",
    "validate_region_bounds",
    # Spatial - Slicing operations
    "compute_regional_slices",
    "compute_centering_slices",
    # Spatial - Dataset operations
    "apply_spatial_selection",
    "crop_dataset_after_centering",
    "assign_spatial_coordinates",
    # Time coordinate functions
    "normalize_time_value",
    "normalize_time_range",
    "read_time_from_file",
    "filter_files_by_time",
]
