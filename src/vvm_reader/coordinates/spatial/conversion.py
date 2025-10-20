"""
VVM Reader Coordinate Conversion

This module provides functions for converting between coordinate-based
and index-based spatial selections.
"""

from typing import Optional, Dict
import numpy as np

from ...core.core_types import IndexRange, CoordinateRange, Region, CoordinateInfo
from ...core.exceptions import ParameterError


# ============================================================================
# Coordinate/Index Conversion
# ============================================================================

def convert_coordinates_to_indices(
    coord_info: CoordinateInfo,
    lon_range: Optional[CoordinateRange] = None,
    lat_range: Optional[CoordinateRange] = None
) -> Dict[str, Optional[IndexRange]]:
    """
    Convert coordinate ranges to index ranges.

    Useful for understanding which indices correspond to given coordinate ranges.

    Args:
        coord_info: Coordinate information
        lon_range: Longitude range to convert
        lat_range: Latitude range to convert

    Returns:
        Dict: Dictionary with 'x_range' and 'y_range' index ranges
    """
    x_indices = None
    y_indices = None

    if lon_range is not None:
        lon_min, lon_max = lon_range
        lon_idx = np.where((coord_info.lon >= lon_min) & (coord_info.lon <= lon_max))[0]
        if lon_idx.size > 0:
            x_indices = (int(lon_idx.min()), int(lon_idx.max()))

    if lat_range is not None:
        lat_min, lat_max = lat_range
        lat_idx = np.where((coord_info.lat >= lat_min) & (coord_info.lat <= lat_max))[0]
        if lat_idx.size > 0:
            y_indices = (int(lat_idx.min()), int(lat_idx.max()))

    return {
        'x_range': x_indices,
        'y_range': y_indices
    }


def convert_indices_to_coordinates(
    coord_info: CoordinateInfo,
    x_range: Optional[IndexRange] = None,
    y_range: Optional[IndexRange] = None
) -> Dict[str, Optional[CoordinateRange]]:
    """
    Convert index ranges to coordinate ranges.

    Useful for understanding which coordinates correspond to given index ranges.

    Args:
        coord_info: Coordinate information
        x_range: X-index range to convert
        y_range: Y-index range to convert

    Returns:
        Dict: Dictionary with 'lon_range' and 'lat_range' coordinate ranges
    """
    lon_coords = None
    lat_coords = None

    if x_range is not None:
        x_min, x_max = x_range
        x_min = max(0, min(x_min, len(coord_info.lon) - 1))
        x_max = max(x_min, min(x_max, len(coord_info.lon) - 1))
        lon_coords = (float(coord_info.lon[x_min]), float(coord_info.lon[x_max]))

    if y_range is not None:
        y_min, y_max = y_range
        y_min = max(0, min(y_min, len(coord_info.lat) - 1))
        y_max = max(y_min, min(y_max, len(coord_info.lat) - 1))
        lat_coords = (float(coord_info.lat[y_min]), float(coord_info.lat[y_max]))

    return {
        'lon_range': lon_coords,
        'lat_range': lat_coords
    }


# ============================================================================
# Validation
# ============================================================================

def validate_region_bounds(region: Region, coord_info: CoordinateInfo) -> None:
    """
    Validate that region bounds are within coordinate ranges.

    Args:
        region: Region specification
        coord_info: Available coordinate information

    Raises:
        ParameterError: If region bounds are invalid
    """
    nx, ny = len(coord_info.lon), len(coord_info.lat)

    # Validate index-based selection
    if region.x_range is not None:
        x_min, x_max = region.x_range
        if x_min < 0 or x_max >= nx:
            raise ParameterError(
                "x_range",
                str(region.x_range),
                f"Outside valid index range: [0, {nx-1}]"
            )

    if region.y_range is not None:
        y_min, y_max = region.y_range
        if y_min < 0 or y_max >= ny:
            raise ParameterError(
                "y_range",
                str(region.y_range),
                f"Outside valid index range: [0, {ny-1}]"
            )

    # Validate coordinate-based selection (only if not overridden by index selection)
    if region.lon_range is not None and region.x_range is None:
        lon_min, lon_max = coord_info.lon.min(), coord_info.lon.max()
        req_min, req_max = region.lon_range

        if req_min < lon_min or req_max > lon_max:
            raise ParameterError(
                "lon_range",
                str(region.lon_range),
                f"Outside available range: [{lon_min:.3f}, {lon_max:.3f}]"
            )

    if region.lat_range is not None and region.y_range is None:
        lat_min, lat_max = coord_info.lat.min(), coord_info.lat.max()
        req_min, req_max = region.lat_range

        if req_min < lat_min or req_max > lat_max:
            raise ParameterError(
                "lat_range",
                str(region.lat_range),
                f"Outside available range: [{lat_min:.3f}, {lat_max:.3f}]"
            )
