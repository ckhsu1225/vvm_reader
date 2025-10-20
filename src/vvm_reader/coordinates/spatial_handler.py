"""
VVM Reader Spatial Coordinate Processing

This module handles spatial coordinate processing including region selection,
slice computation, and coordinate system validation.
"""

from typing import Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import xarray as xr

from ..core.config import LON_DIM, LAT_DIM, TOPO_VAR, VERTICAL_DIM, get_simulation_paths
from ..core.core_types import IndexRange, CoordinateRange, Region, CoordinateInfo, SliceInfo
from ..core.exceptions import (
    CoordinateError, ParameterError,
    validate_required_file
)

# ============================================================================
# TOPO.nc Loading and Validation
# ============================================================================

def load_topo_dataset(sim_dir: Path, engine: Optional[str] = None) -> xr.Dataset:
    """
    Load and validate TOPO.nc dataset.
    
    Args:
        sim_dir: Simulation directory path
        engine: xarray backend engine
        
    Returns:
        xr.Dataset: Loaded TOPO dataset
        
    Raises:
        RequiredFileNotFoundError: If TOPO.nc not found
        CoordinateError: If required coordinates are missing
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
    
    Args:
        terrain_mask: 3D terrain mask array from TOPO.nc
    
    Returns:
        xr.DataArray: Surface topography levels
    """
    if terrain_mask is None:
        raise CoordinateError("TOPO.nc", "Terrain mask variable 'mask' is required")

    if VERTICAL_DIM not in terrain_mask.dims or terrain_mask.ndim != 3:
        raise CoordinateError("TOPO.nc", "Terrain mask must be 3D with a vertical dimension")

    mask_bool = terrain_mask.astype(bool)
    subsurface_levels = (~mask_bool).sum(dim=VERTICAL_DIM)
    return subsurface_levels

def extract_terrain_mask(topo_ds: xr.Dataset) -> xr.DataArray:
    """
    Extract the 3D terrain mask from ``TOPO.nc``.
    
    Args:
        topo_ds: TOPO dataset
    
    Returns:
        xr.DataArray: 3D terrain mask
    """
    if "mask" not in topo_ds:
        raise CoordinateError("TOPO.nc", "Terrain mask variable 'mask' is required")

    mask = topo_ds["mask"]
    if VERTICAL_DIM not in mask.dims or mask.ndim != 3:
        raise CoordinateError("TOPO.nc", "Terrain mask must be 3D with a vertical dimension")

    mask_bool = mask.astype(bool)
    mask_bool.name = "terrain_mask"
    return mask_bool

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
    needs_u_halo: bool = False,
    needs_v_halo: bool = False
) -> Tuple[SliceInfo, Dict[str, int]]:
    """
    Compute extended slices for wind centering with halo regions.
    
    Args:
        original_slice_info: Original slice information
        needs_u_halo: Whether u-wind needs x-direction halo
        needs_v_halo: Whether v-wind needs y-direction halo
        
    Returns:
        Tuple[SliceInfo, Dict[str, int]]: Extended slices and crop offsets
    """
    x_slice = original_slice_info.x_slice
    y_slice = original_slice_info.y_slice
    
    crop_offsets = {"x": 0, "y": 0}
    
    # Extend x-slice for u-wind centering if needed
    if (needs_u_halo and not original_slice_info.periodic_x and 
        x_slice.start is not None and x_slice.start > 0):
        x_slice = slice(x_slice.start - 1, x_slice.stop)
        crop_offsets["x"] = 1
    
    # Extend y-slice for v-wind centering if needed  
    if (needs_v_halo and not original_slice_info.periodic_y and
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

def apply_spatial_selection(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo
) -> xr.Dataset:
    """
    Apply spatial selection to a dataset.
    
    Args:
        dataset: Input dataset
        coord_info: Coordinate information
        slice_info: Slice information
        
    Returns:
        xr.Dataset: Spatially selected dataset
    """
    indexers = {}
    
    if coord_info.x_dim in dataset.sizes:
        indexers[coord_info.x_dim] = slice_info.x_slice
    
    if coord_info.y_dim in dataset.sizes:
        indexers[coord_info.y_dim] = slice_info.y_slice
    
    if indexers:
        dataset = dataset.isel(indexers)
    
    return dataset

def crop_dataset_after_centering(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    crop_offsets: Dict[str, int],
    target_shape: Dict[str, int]
) -> xr.Dataset:
    """
    Crop dataset back to original selection after wind centering.
    
    Args:
        dataset: Dataset after centering operations
        coord_info: Coordinate information
        crop_offsets: Cropping offsets from centering
        target_shape: Target shape for each dimension
        
    Returns:
        xr.Dataset: Cropped dataset
    """
    indexers = {}
    
    if crop_offsets["x"] > 0 and coord_info.x_dim in dataset.sizes:
        nx = target_shape.get(coord_info.x_dim, dataset.sizes[coord_info.x_dim])
        indexers[coord_info.x_dim] = slice(crop_offsets["x"], crop_offsets["x"] + nx)
    
    if crop_offsets["y"] > 0 and coord_info.y_dim in dataset.sizes:
        ny = target_shape.get(coord_info.y_dim, dataset.sizes[coord_info.y_dim])
        indexers[coord_info.y_dim] = slice(crop_offsets["y"], crop_offsets["y"] + ny)
    
    if indexers:
        dataset = dataset.isel(indexers)
    
    return dataset

# ============================================================================
# Coordinate Assignment
# ============================================================================

def assign_spatial_coordinates(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo
) -> xr.Dataset:
    """
    Assign lon/lat coordinates to dataset.
    
    Args:
        dataset: Input dataset
        coord_info: Full coordinate information
        slice_info: Applied slice information
        
    Returns:
        xr.Dataset: Dataset with assigned coordinates
    """
    # Extract subsets of lon/lat based on slices
    lon_subset = coord_info.lon[slice_info.x_slice]
    lat_subset = coord_info.lat[slice_info.y_slice]
    
    # Create coordinate arrays with proper dimension names
    lon_coord = xr.DataArray(lon_subset, dims=[coord_info.x_dim])
    if coord_info.lon_attrs:
        lon_coord.attrs.update(coord_info.lon_attrs)
    
    lat_coord = xr.DataArray(lat_subset, dims=[coord_info.y_dim])
    if coord_info.lat_attrs:
        lat_coord.attrs.update(coord_info.lat_attrs)
    
    return dataset.assign_coords({
        LON_DIM: lon_coord,
        LAT_DIM: lat_coord
    })

# ============================================================================
# Spatial Information and Utilities
# ============================================================================

def get_spatial_info(coord_info: CoordinateInfo, slice_info: SliceInfo) -> Dict:
    """
    Get spatial domain information.
    
    Args:
        coord_info: Coordinate information
        slice_info: Slice information
        
    Returns:
        Dict: Spatial domain information
    """
    lon_subset = coord_info.lon[slice_info.x_slice]
    lat_subset = coord_info.lat[slice_info.y_slice]
    
    return {
        'nx_total': len(coord_info.lon),
        'ny_total': len(coord_info.lat),
        'nx_selected': len(lon_subset),
        'ny_selected': len(lat_subset),
        'lon_range': (float(lon_subset.min()), float(lon_subset.max())) if len(lon_subset) > 0 else None,
        'lat_range': (float(lat_subset.min()), float(lat_subset.max())) if len(lat_subset) > 0 else None,
        'periodic_x': slice_info.periodic_x,
        'periodic_y': slice_info.periodic_y,
        'x_slice': slice_info.x_slice,
        'y_slice': slice_info.y_slice,
        'x_indices': (slice_info.x_slice.start, slice_info.x_slice.stop - 1) if slice_info.x_slice.start is not None else None,
        'y_indices': (slice_info.y_slice.start, slice_info.y_slice.stop - 1) if slice_info.y_slice.start is not None else None,
    }

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

def compute_resolution_info(coord_info: CoordinateInfo) -> Dict:
    """
    Compute grid resolution information.
    
    Args:
        coord_info: Coordinate information
        
    Returns:
        Dict: Resolution information
    """
    lon_diff = np.diff(coord_info.lon)
    lat_diff = np.diff(coord_info.lat)
    
    return {
        'dx_mean': float(np.mean(lon_diff)),
        'dx_min': float(np.min(lon_diff)),
        'dx_max': float(np.max(lon_diff)),
        'dy_mean': float(np.mean(lat_diff)), 
        'dy_min': float(np.min(lat_diff)),
        'dy_max': float(np.max(lat_diff)),
        'uniform_x': bool(np.allclose(lon_diff, lon_diff[0])),
        'uniform_y': bool(np.allclose(lat_diff, lat_diff[0]))
    }
