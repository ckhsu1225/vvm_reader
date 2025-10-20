"""
VVM Reader Spatial Slicing

This module handles computation of spatial slices for regional selection
and wind centering operations.
"""

from typing import Tuple, Dict
import numpy as np

from ...core.core_types import Region, CoordinateInfo, SliceInfo


# ============================================================================
# Region Selection and Slice Computation
# ============================================================================

def compute_regional_slices(
    coord_info: CoordinateInfo,
    region: Region
) -> SliceInfo:
    """
    Compute spatial slices for regional selection.

    Supports both coordinate-based (lon/lat) and index-based (x/y) selection.
    Index-based selection takes priority when both are specified.

    Args:
        coord_info: Coordinate information
        region: Regional selection parameters

    Returns:
        SliceInfo: Computed slice information
    """
    lon, lat = coord_info.lon, coord_info.lat
    nx, ny = len(lon), len(lat)

    # Default to full domain
    x_slice = slice(0, nx)
    y_slice = slice(0, ny)

    # Handle X-direction selection (index takes priority over coordinate)
    if region.x_range is not None:
        # Index-based selection
        x_min, x_max = region.x_range
        x_min = max(0, min(x_min, nx - 1))
        x_max = max(x_min, min(x_max, nx - 1))
        x_slice = slice(x_min, x_max + 1)

    elif region.lon_range is not None:
        # Coordinate-based selection
        lon_min, lon_max = region.lon_range
        lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        if lon_indices.size > 0:
            x_slice = slice(int(lon_indices.min()), int(lon_indices.max()) + 1)
        else:
            x_slice = slice(0, 0)  # Empty slice

    # Handle Y-direction selection (index takes priority over coordinate)
    if region.y_range is not None:
        # Index-based selection
        y_min, y_max = region.y_range
        y_min = max(0, min(y_min, ny - 1))
        y_max = max(y_min, min(y_max, ny - 1))
        y_slice = slice(y_min, y_max + 1)

    elif region.lat_range is not None:
        # Coordinate-based selection
        lat_min, lat_max = region.lat_range
        lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        if lat_indices.size > 0:
            y_slice = slice(int(lat_indices.min()), int(lat_indices.max()) + 1)
        else:
            y_slice = slice(0, 0)  # Empty slice

    # Determine periodicity (full domain selection)
    periodic_x = (x_slice.start == 0 and x_slice.stop == nx and
                  (x_slice.step is None or x_slice.step == 1))
    periodic_y = (y_slice.start == 0 and y_slice.stop == ny and
                  (y_slice.step is None or y_slice.step == 1))

    return SliceInfo(
        x_slice=x_slice,
        y_slice=y_slice,
        periodic_x=periodic_x,
        periodic_y=periodic_y
    )


def compute_centering_slices(
    original_slice_info: SliceInfo,
    needs_x_halo: bool = False,
    needs_y_halo: bool = False
) -> Tuple[SliceInfo, Dict[str, int]]:
    """
    Compute extended slices for wind centering with halo regions.

    Args:
        original_slice_info: Original slice information
        needs_x_halo: Whether needs x-direction halo
        needs_y_halo: Whether needs y-direction halo

    Returns:
        Tuple[SliceInfo, Dict[str, int]]: Extended slices and crop offsets
    """
    x_slice = original_slice_info.x_slice
    y_slice = original_slice_info.y_slice

    crop_offsets = {"x": 0, "y": 0}

    # Extend x-slice for centering if needed
    if (needs_x_halo and not original_slice_info.periodic_x and
        x_slice.start is not None and x_slice.start > 0):
        x_slice = slice(x_slice.start - 1, x_slice.stop)
        crop_offsets["x"] = 1

    # Extend y-slice for centering if needed
    if (needs_y_halo and not original_slice_info.periodic_y and
        y_slice.start is not None and y_slice.start > 0):
        y_slice = slice(y_slice.start - 1, y_slice.stop)
        crop_offsets["y"] = 1

    extended_slice_info = SliceInfo(
        x_slice=x_slice,
        y_slice=y_slice,
        periodic_x=original_slice_info.periodic_x,
        periodic_y=original_slice_info.periodic_y
    )

    return extended_slice_info, crop_offsets
