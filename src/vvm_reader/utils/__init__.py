"""
VVM Reader Utilities

This package provides utility functions for simulation information queries,
coordinate conversions, and manifest creation.
"""

# Information functions
from .info import (
    list_available_simulations,
    get_simulation_info,
    get_coordinate_info,
    get_terrain_info,
    get_vertical_info,
    get_spatial_info,
)

# Conversion functions
from .conversion import (
    convert_coordinates_to_indices,
    convert_indices_to_coordinates,
    convert_time_to_indices,
    convert_heights_to_indices,
    create_variable_manifest,
)

__all__ = [
    # Information functions
    "list_available_simulations",
    "get_simulation_info",
    "get_coordinate_info",
    "get_terrain_info",
    "get_vertical_info",
    "get_spatial_info",
    # Conversion functions
    "convert_coordinates_to_indices",
    "convert_indices_to_coordinates",
    "convert_time_to_indices",
    "convert_heights_to_indices",
    "create_variable_manifest",
]
