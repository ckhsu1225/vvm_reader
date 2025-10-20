"""
VVM Reader Spatial Coordinate Handling

This package provides spatial coordinate processing functionality including
TOPO.nc loading, coordinate conversion, slice computation, and dataset operations.
"""

# Loader functions
from .loader import (
    load_topo_dataset,
    extract_coordinates_from_topo,
    compute_surface_topo_levels,
    extract_terrain_mask,
    extract_terrain_height,
)

# Conversion functions
from .conversion import (
    convert_coordinates_to_indices,
    convert_indices_to_coordinates,
    validate_region_bounds,
)

# Slicing functions
from .slicing import (
    compute_regional_slices,
    compute_centering_slices,
)

# Dataset operations
from .operations import (
    apply_spatial_selection,
    crop_dataset_after_centering,
    assign_spatial_coordinates,
)

__all__ = [
    # Loader functions
    "load_topo_dataset",
    "extract_coordinates_from_topo",
    "compute_surface_topo_levels",
    "extract_terrain_mask",
    "extract_terrain_height",
    # Conversion functions
    "convert_coordinates_to_indices",
    "convert_indices_to_coordinates",
    "validate_region_bounds",
    # Slicing functions
    "compute_regional_slices",
    "compute_centering_slices",
    # Dataset operations
    "apply_spatial_selection",
    "crop_dataset_after_centering",
    "assign_spatial_coordinates",
]
