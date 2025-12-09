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
from ...core.logging_config import get_logger


# ============================================================================
# TOPO.nc Loading and Validation
# ============================================================================

def _find_topo_variable(ds: xr.Dataset) -> Optional[str]:
    """
    Find the topography variable name in the dataset.

    Checks for common case variations of the topo variable name.

    Args:
        ds: Dataset to search

    Returns:
        Variable name if found, None otherwise
    """
    # Check for the configured name first
    if TOPO_VAR in ds:
        return TOPO_VAR
    
    # Check for common case variations
    topo_variations = ['TOPO', 'Topo', 'topo']
    for var_name in topo_variations:
        if var_name in ds:
            return var_name
    
    return None


def load_topo_dataset(sim_dir: Path, engine: Optional[str] = None) -> xr.Dataset:
    """
    Load and validate TOPO.nc dataset with caching.

    Args:
        sim_dir: Simulation directory path
        engine: xarray backend engine

    Returns:
        xr.Dataset: Loaded TOPO dataset with standardized variable names

    Raises:
        RequiredFileNotFoundError: If TOPO.nc not found
        CoordinateError: If required coordinates are missing

    """
    topo_path = get_simulation_paths(sim_dir)['topo']
    validate_required_file(topo_path, "TOPO.nc")

    try:
        topo_ds = xr.open_dataset(topo_path, engine=engine, decode_times=False)
    except Exception as e:
        raise CoordinateError("TOPO.nc", f"Failed to open file: {e}")

    # Validate required coordinate variables
    required_coord_vars = [LON_DIM, LAT_DIM]
    missing_vars = [var for var in required_coord_vars if var not in topo_ds]
    
    # Find the topo variable (handles case variations)
    actual_topo_var = _find_topo_variable(topo_ds)
    if actual_topo_var is None:
        missing_vars.append(TOPO_VAR)
    
    if missing_vars:
        raise CoordinateError("TOPO.nc", f"Missing required variables: {missing_vars}")

    # Rename topo variable to standard name if needed
    if actual_topo_var is not None and actual_topo_var != TOPO_VAR:
        topo_ds = topo_ds.rename({actual_topo_var: TOPO_VAR})

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

    return topo
    # Apply the standard masking rule: convert ocean (0) -> land (1)
    # This ensures k <= topo masking works correctly
    # return _apply_topo_masking_rule(topo)


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

    If the 'height' variable is missing (common in simulations without terrain),
    returns a zero-filled array with a warning.

    Args:
        topo_ds: TOPO dataset

    Returns:
        xr.DataArray: Terrain height in meters (lat, lon)

    Raises:
        CoordinateError: If topo variable is missing or dimensions are invalid
    """
    import numpy as np
    
    logger = get_logger('coordinates.spatial')
    
    if "height" not in topo_ds:
        # Create zero-height array for simulations without terrain
        logger.warning(
            "TOPO.nc missing 'height' variable. "
            "Assuming no terrain (height=0). This is common for ocean/flat simulations."
        )
        
        # Get dimensions from topo variable (handle case variations)
        actual_topo_var = _find_topo_variable(topo_ds)
        if actual_topo_var is None:
            raise CoordinateError("TOPO.nc", f"Required variable '{TOPO_VAR}' (or case variation) not found")
        
        topo = topo_ds[actual_topo_var]
        
        # Create zero-filled array with same shape and coordinates
        height_m = xr.DataArray(
            np.zeros(topo.shape, dtype=np.float32),
            dims=topo.dims,
            coords=topo.coords,
            attrs={
                'units': 'm',
                'long_name': 'terrain height',
                'standard_name': 'surface_altitude',
                '_note': 'Zero-filled: original height variable missing from TOPO.nc'
            },
            name='terrain_height'
        )
        return height_m

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
