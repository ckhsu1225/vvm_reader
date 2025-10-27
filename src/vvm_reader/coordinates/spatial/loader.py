"""
VVM Reader Spatial Coordinate Loader

This module handles loading and extracting coordinate information from TOPO.nc.
"""

from typing import Optional
from pathlib import Path
import xarray as xr

from ...core.config import LON_DIM, LAT_DIM, TOPO_VAR, get_simulation_paths
from ...core.core_types import CoordinateInfo
from ...core.exceptions import CoordinateError, validate_required_file


# ============================================================================
# TOPO.nc Loading and Validation
# ============================================================================

def load_topo_dataset(sim_dir: Path, engine: Optional[str] = None) -> xr.Dataset:
    """
    Load and validate TOPO.nc dataset with caching.

    This function caches up to 32 different simulations to avoid repeatedly
    reading the same TOPO.nc file. This significantly improves performance
    when querying multiple types of information from the same simulation.

    Args:
        sim_dir: Simulation directory path
        engine: xarray backend engine

    Returns:
        xr.Dataset: Loaded TOPO dataset

    Raises:
        RequiredFileNotFoundError: If TOPO.nc not found
        CoordinateError: If required coordinates are missing

    Note:
        Cache can be cleared with: load_topo_dataset.cache_clear()
    """
    topo_path = get_simulation_paths(sim_dir)['topo']
    validate_required_file(topo_path, "TOPO.nc")

    try:
        topo_ds = xr.open_dataset(topo_path, engine=engine)
    except Exception as e:
        raise CoordinateError("TOPO.nc", f"Failed to open file: {e}")

    # Validate required variables
    required_vars = [LON_DIM, LAT_DIM, TOPO_VAR]
    missing_vars = [var for var in required_vars if var not in topo_ds]
    if missing_vars:
        raise CoordinateError("TOPO.nc", f"Missing required variables: {missing_vars}")

    return topo_ds


def extract_coordinates_from_topo(topo_ds: xr.Dataset) -> CoordinateInfo:
    """
    Extract coordinate information from TOPO dataset.

    Args:
        topo_ds: TOPO dataset

    Returns:
        CoordinateInfo: Extracted coordinate information

    Raises:
        CoordinateError: If coordinates are invalid
    """
    lon = topo_ds[LON_DIM]
    lat = topo_ds[LAT_DIM]

    # Validate coordinate dimensions
    if lon.ndim != 1:
        raise CoordinateError("longitude", f"Expected 1D, got {lon.ndim}D")
    if lat.ndim != 1:
        raise CoordinateError("latitude", f"Expected 1D, got {lat.ndim}D")

    # Get dimension names
    x_dim = lon.dims[0]
    y_dim = lat.dims[0]

    return CoordinateInfo(
        lon=lon.values,
        lat=lat.values,
        x_dim=x_dim,
        y_dim=y_dim,
        lon_attrs=dict(getattr(lon, "attrs", {})),
        lat_attrs=dict(getattr(lat, "attrs", {}))
    )


def prepare_topo_data(topo_ds: xr.Dataset) -> xr.DataArray:
    """
    Prepare topography data from TOPO dataset.

    This function extracts the 2D topo array and applies the ocean->land masking rule.
    The topo array contains the vertical index (k) of the terrain top at each (x,y) point.

    Performance note: Returns a 2D array instead of a 3D mask, which is much more
    efficient for memory usage and I/O. The 3D mask is dynamically generated when
    needed via broadcasting in apply_terrain_mask().

    Args:
        topo_ds: TOPO dataset

    Returns:
        xr.DataArray: 2D topography data with terrain top indices
    """
    if TOPO_VAR not in topo_ds:
        raise CoordinateError("TOPO.nc", f"Topography variable '{TOPO_VAR}' is required")

    topo = topo_ds[TOPO_VAR]

    # Validate dimensions
    if topo.ndim != 2:
        raise CoordinateError("topo", f"Expected 2D, got {topo.ndim}D")

    # Apply the standard masking rule: convert ocean (0) -> land (1)
    # This ensures k <= topo masking works correctly
    return _apply_topo_masking_rule(topo)


def _apply_topo_masking_rule(topo: xr.DataArray) -> xr.DataArray:
    """
    Apply the standard topography masking rule.

    Converts ocean (0) -> land (1) to match the terrain masking rule k <= topo.

    Args:
        topo: Raw topography data

    Returns:
        xr.DataArray: Processed topography data
    """
    from ...core.config import MIN_LAND_TOPO_VALUE
    return topo.where(topo >= MIN_LAND_TOPO_VALUE, MIN_LAND_TOPO_VALUE)


def extract_terrain_height(topo_ds: xr.Dataset) -> xr.DataArray:
    """
    Extract terrain height from TOPO.nc and convert from km to meters.

    Args:
        topo_ds: TOPO dataset

    Returns:
        xr.DataArray: Terrain height in meters (lat, lon)

    Raises:
        CoordinateError: If height variable not found or invalid
    """
    if "height" not in topo_ds:
        raise CoordinateError("TOPO.nc", "Terrain height variable 'height' is required")

    height = topo_ds["height"]

    # Validate dimensions
    if height.ndim != 2:
        raise CoordinateError("height", f"Expected 2D (lat, lon), got {height.ndim}D")

    # Convert from km to meters
    height_m = height * 1000.0

    # Update attributes
    height_m.attrs = dict(height.attrs) if hasattr(height, 'attrs') else {}
    height_m.attrs['units'] = 'm'
    height_m.attrs['long_name'] = 'terrain height'
    height_m.attrs['standard_name'] = 'surface_altitude'
    height_m.name = 'terrain_height'

    return height_m
