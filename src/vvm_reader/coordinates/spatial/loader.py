"""
VVM Reader Spatial Coordinate Loader

This module handles loading and extracting coordinate information from TOPO.nc.
"""

from functools import lru_cache
from typing import Optional
from pathlib import Path
import xarray as xr

from ...core.config import LON_DIM, LAT_DIM, TOPO_VAR, VERTICAL_DIM, get_simulation_paths
from ...core.core_types import CoordinateInfo
from ...core.exceptions import CoordinateError, validate_required_file


# ============================================================================
# TOPO.nc Loading and Validation
# ============================================================================

@lru_cache(maxsize=32)
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
    required_vars = [LON_DIM, LAT_DIM, TOPO_VAR, "mask"]
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


def compute_surface_topo_levels(terrain_mask: xr.DataArray) -> xr.DataArray:
    """
    Compute the surface topography levels from the terrain mask.

    Performance note: This function keeps the computation lazy. The actual
    data loading and computation will be deferred until the surface_level
    is accessed, and only after spatial slicing has been applied.

    Args:
        terrain_mask: 3D terrain mask array from TOPO.nc (may be lazy)

    Returns:
        xr.DataArray: Surface topography levels (lazy computation)
    """
    if terrain_mask is None:
        raise CoordinateError("TOPO.nc", "Terrain mask variable 'mask' is required")

    if VERTICAL_DIM not in terrain_mask.dims or terrain_mask.ndim != 3:
        raise CoordinateError("TOPO.nc", "Terrain mask must be 3D with a vertical dimension")

    # Keep computation lazy - astype(bool) and sum() will be deferred
    # This allows xarray to optimize the computation after spatial slicing
    mask_bool = terrain_mask.astype(bool)
    surface_level = (~mask_bool).sum(dim=VERTICAL_DIM)
    surface_level.name = "surface_level"
    return surface_level


def extract_terrain_mask(topo_ds: xr.Dataset) -> xr.DataArray:
    """
    Extract the 3D terrain mask from TOPO.nc.

    Performance note: Returns the mask without type conversion to keep it lazy.
    The astype(bool) conversion will be deferred until after spatial slicing,
    which significantly reduces memory and loading time.

    Args:
        topo_ds: TOPO dataset

    Returns:
        xr.DataArray: 3D terrain mask (lazy, not yet converted to bool)
    """
    if "mask" not in topo_ds:
        raise CoordinateError("TOPO.nc", "Terrain mask variable 'mask' is required")

    # Keep mask lazy - don't convert to bool yet
    # This allows xarray to defer loading until after spatial/vertical slicing
    mask = topo_ds["mask"]

    if VERTICAL_DIM not in mask.dims or mask.ndim != 3:
        raise CoordinateError("TOPO.nc", "Terrain mask must be 3D with a vertical dimension")
    return mask


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
