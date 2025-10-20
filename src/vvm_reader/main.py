"""
VVM Reader Main Interface

This module provides the main interface functions for loading VVM datasets
using the new structured API design.
"""

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Tuple, List
import numpy as np
import xarray as xr

from .core.config import TOPO_VAR
from .core.core_types import (
    LoadParameters, Region, TimeSelection, VerticalSelection, ProcessingOptions
)
from .io.dataset_loader import load_vvm_dataset

# ============================================================================
# Main API Function
# ============================================================================

def open_vvm_dataset(
    sim_dir: Union[str, Path],
    *,
    groups: Optional[Sequence[str]] = None,
    variables: Optional[Sequence[str]] = None,
    region: Optional[Region] = None,
    time_selection: Optional[TimeSelection] = None,
    vertical_selection: Optional[VerticalSelection] = None,
    processing_options: Optional[ProcessingOptions] = None,
    var_manifest: Optional[Union[str, Path, Dict]] = None,
) -> xr.Dataset:
    """
    Load VVM dataset with structured parameters.
    
    This is the main interface for loading VVM data using structured parameter objects
    for better organization and type safety.
    
    Args:
        sim_dir: Path to simulation directory
        groups: Output groups to load (e.g., ["L.Dynamic", "L.Radiation"])
        variables: Specific variables to load
        region: Spatial region selection
        time_selection: Time selection parameters
        vertical_selection: Vertical level selection parameters
        processing_options: Data processing options
        var_manifest: Variable manifest (path, dict, or None for auto-load)
        
    Returns:
        xr.Dataset: Loaded and processed VVM dataset
        
    Examples:
        # Load all data
        >>> ds = open_vvm_dataset("/path/to/sim")
        
        # Load specific variables with spatial/temporal selection
        >>> region = Region(lon_range=(120, 122), lat_range=(23, 25))
        >>> time_sel = TimeSelection(time_index_range=(0, 36))
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["u", "v", "w", "th"],
        ...     region=region,
        ...     time_selection=time_sel
        ... )
        
        # Load with index-based selection
        >>> region = Region(x_range=(100, 200), y_range=(50, 150))
        >>> vert_sel = VerticalSelection(index_range=(5, 25))
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     region=region,
        ...     vertical_selection=vert_sel
        ... )
    """
    # Create parameter object with defaults
    params = LoadParameters(
        groups=groups,
        variables=variables,
        region=region or Region(),
        time_selection=time_selection or TimeSelection(),
        vertical_selection=vertical_selection or VerticalSelection(),
        processing_options=processing_options or ProcessingOptions(),
        var_manifest=var_manifest
    )
    
    return load_vvm_dataset(Path(sim_dir), params)

# ============================================================================
# Convenience Functions
# ============================================================================

def quick_load(
    sim_dir: Union[str, Path],
    variables: Optional[Sequence[str]] = None,
    time_steps: Optional[int] = None,
    **kwargs
) -> xr.Dataset:
    """
    Quick load function for common use cases.
    
    Args:
        sim_dir: Path to simulation directory
        variables: Variables to load (None for all)
        time_steps: Number of time steps to load from start
        **kwargs: Additional arguments passed to open_vvm_dataset
        
    Returns:
        xr.Dataset: Loaded dataset
        
    Examples:
        # Load first 10 time steps of u, v, w
        >>> ds = quick_load("/path/to/sim", ["u", "v", "w"], time_steps=10)
        
        # Load all thermodynamic variables
        >>> ds = quick_load("/path/to/sim", groups=["L.Thermodynamic"])
    """
    time_selection = None
    if time_steps is not None:
        time_selection = TimeSelection(time_index_range=(0, time_steps - 1))
    
    return open_vvm_dataset(
        sim_dir=sim_dir,
        variables=variables,
        time_selection=time_selection,
        **kwargs
    )

def load_surface_data(
    sim_dir: Union[str, Path],
    variables: Optional[Sequence[str]] = None,
    auto_vertical_range: bool = True,
    **kwargs
) -> xr.Dataset:
    """
    Load only surface data with automatic vertical range detection.
    
    Args:
        sim_dir: Path to simulation directory
        variables: Variables to load
        auto_vertical_range: Automatically determine vertical range from topography
        **kwargs: Additional arguments passed to open_vvm_dataset
        
    Returns:
        xr.Dataset: Surface dataset without vertical dimension
    """
    vertical_selection = VerticalSelection(
        surface_nearest=True,
        surface_only=True
    )
    
    # Automatically determine vertical range from topography
    if auto_vertical_range and 'vertical_selection' not in kwargs:
        try:
            region = kwargs.get('region')
            terrain_info = get_terrain_info(sim_dir, region)
            vertical_selection.index_range = (0, terrain_info['max_level'])
            
        except Exception as e:
            print(f"Warning: Could not auto-detect vertical range ({e}), loading all levels")
    
    return open_vvm_dataset(
        sim_dir=sim_dir,
        variables=variables,
        vertical_selection=vertical_selection,
        **kwargs
    )

def load_region(
    sim_dir: Union[str, Path],
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
    variables: Optional[Sequence[str]] = None,
    **kwargs
) -> xr.Dataset:
    """
    Load data for a specific geographical region.
    
    Supports both coordinate-based (lon/lat) and index-based (x/y) selection.
    Index-based selection takes priority when both are specified.
    
    Args:
        sim_dir: Path to simulation directory
        lon_range: Longitude range (min_lon, max_lon) in degrees
        lat_range: Latitude range (min_lat, max_lat) in degrees
        x_range: X-index range (min_x, max_x) - takes priority over lon_range
        y_range: Y-index range (min_y, max_y) - takes priority over lat_range
        variables: Variables to load
        **kwargs: Additional arguments passed to open_vvm_dataset
        
    Returns:
        xr.Dataset: Regional dataset
        
    Examples:
        # Coordinate-based selection
        >>> ds = load_region("/path/to/sim", lon_range=(120, 122), lat_range=(23, 25))
        
        # Index-based selection (more precise)
        >>> ds = load_region("/path/to/sim", x_range=(100, 200), y_range=(50, 150))
    """
    region = Region(
        lon_range=lon_range,
        lat_range=lat_range,
        x_range=x_range,
        y_range=y_range
    )
    
    return open_vvm_dataset(
        sim_dir=sim_dir,
        variables=variables,
        region=region,
        **kwargs
    )

def load_indices(
    sim_dir: Union[str, Path],
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    variables: Optional[Sequence[str]] = None,
    **kwargs
) -> xr.Dataset:
    """
    Load data using index-based selection.
    
    Args:
        sim_dir: Path to simulation directory
        x_range: X-index range (min_x, max_x)
        y_range: Y-index range (min_y, max_y)
        variables: Variables to load
        **kwargs: Additional arguments passed to open_vvm_dataset
        
    Returns:
        xr.Dataset: Dataset with index-based selection
        
    Examples:
        # Load a 100x100 subdomain starting from index (50, 50)
        >>> ds = load_indices("/path/to/sim", (50, 149), (50, 149), ["u", "v", "w"])
    """
    region = Region(x_range=x_range, y_range=y_range)
    
    return open_vvm_dataset(
        sim_dir=sim_dir,
        variables=variables,
        region=region,
        **kwargs
    )

def load_time_series(
    sim_dir: Union[str, Path],
    variables: Sequence[str],
    time_range: Optional[Tuple] = None,
    spatial_mean: bool = False,
    **kwargs
) -> xr.Dataset:
    """
    Load time series data, optionally computing spatial means.
    
    Args:
        sim_dir: Path to simulation directory
        variables: Variables to load
        time_range: Time range (start_time, end_time)
        spatial_mean: Whether to compute spatial means
        **kwargs: Additional arguments passed to open_vvm_dataset
        
    Returns:
        xr.Dataset: Time series dataset
    """
    time_selection = None
    if time_range is not None:
        time_selection = TimeSelection(time_range=time_range)
    
    ds = open_vvm_dataset(
        sim_dir=sim_dir,
        variables=variables,
        time_selection=time_selection,
        **kwargs
    )
    
    if spatial_mean:
        # Compute spatial means, preserving time and vertical dimensions
        spatial_dims = []
        if 'x' in ds.sizes:
            spatial_dims.append('x')
        if 'y' in ds.sizes:
            spatial_dims.append('y')
        
        if spatial_dims:
            ds = ds.mean(dim=spatial_dims, keep_attrs=True)
    
    return ds

# ============================================================================
# Utility Functions
# ============================================================================

def list_available_simulations(parent_dir: Union[str, Path]) -> List[Path]:
    """
    List all available VVM simulations in a directory.
    
    Args:
        parent_dir: Parent directory to search
        
    Returns:
        List[Path]: List of simulation directory paths
    """
    from .io.file_utils import find_simulation_directories
    return find_simulation_directories(Path(parent_dir))

def get_simulation_info(sim_dir: Union[str, Path]) -> Dict:
    """
    Get information about a simulation directory.
    
    Args:
        sim_dir: Simulation directory path
        
    Returns:
        Dict: Simulation information
    """
    from .io.file_utils import get_simulation_info as _get_info
    return _get_info(Path(sim_dir))

def get_terrain_info(sim_dir: Union[str, Path], region: Optional[Region] = None) -> Dict:
    """
    Get terrain information for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        region: Optional region to analyze (if None, analyzes entire domain)
        
    Returns:
        Dict: Terrain information including min/max levels and statistics
        
    Examples:
        >>> terrain_info = get_terrain_info("/path/to/sim")
        >>> print(f"Terrain levels: {terrain_info['min_level']} to {terrain_info['max_level']}")
        >>> print(f"Recommended vertical range: 0 to {terrain_info['recommended_max_level']}")
        
        >>> # Check terrain in a specific region
        >>> region = Region(lon_range=(120, 122), lat_range=(23, 25))
        >>> terrain_info = get_terrain_info("/path/to/sim", region=region)
    """
    try:
        from .coordinates.spatial_handler import (
            load_topo_dataset, extract_coordinates_from_topo,
            compute_regional_slices
        )
        
        # Load topography data
        topo_ds = load_topo_dataset(Path(sim_dir))
        coord_info = extract_coordinates_from_topo(topo_ds)
        topo_data = topo_ds[TOPO_VAR]
        
        # Apply regional selection if specified
        if region is not None and region.has_selection:
            slice_info = compute_regional_slices(coord_info, region)
            
            indexers = {}
            if coord_info.x_dim in topo_data.sizes:
                indexers[coord_info.x_dim] = slice_info.x_slice
            if coord_info.y_dim in topo_data.sizes:
                indexers[coord_info.y_dim] = slice_info.y_slice
            
            if indexers:
                topo_data = topo_data.isel(indexers)
        
        # Compute terrain statistics
        topo_values = topo_data.values
        
        info = {
            'min_level': int(topo_values.min()),
            'max_level': int(topo_values.max()),
            'mean_level': float(topo_values.mean()),
            'ocean_points': int(np.sum(topo_values == 0)),
            'land_points': int(np.sum(topo_values >= 1)),
            'total_points': int(topo_values.size),
        }
        
        info['ocean_fraction'] = info['ocean_points'] / info['total_points']
        info['land_fraction'] = info['land_points'] / info['total_points']
        
        topo_ds.close()
        return info
        
    except Exception as e:
        raise ValueError(f"Failed to get terrain info: {e}")

def get_coordinate_info(sim_dir: Union[str, Path]) -> Dict:
    """
    Get coordinate system information for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        
    Returns:
        Dict: Coordinate information including ranges and resolution
        
    Examples:
        >>> info = get_coordinate_info("/path/to/sim")
        >>> print(f"Domain: {info['lon_range']} x {info['lat_range']}")
        >>> print(f"Grid: {info['nx']} x {info['ny']}")
        >>> print(f"Resolution: {info['dx_mean']:.3f} x {info['dy_mean']:.3f}")
    """
    from .coordinates.spatial_handler import (
        load_topo_dataset, extract_coordinates_from_topo, compute_resolution_info
    )
    
    try:
        topo_ds = load_topo_dataset(Path(sim_dir))
        coord_info = extract_coordinates_from_topo(topo_ds)
        resolution_info = compute_resolution_info(coord_info)
        topo_ds.close()
        
        return {
            'nx': len(coord_info.lon),
            'ny': len(coord_info.lat),
            'lon_range': (float(coord_info.lon.min()), float(coord_info.lon.max())),
            'lat_range': (float(coord_info.lat.min()), float(coord_info.lat.max())),
            'x_dim': coord_info.x_dim,
            'y_dim': coord_info.y_dim,
            **resolution_info
        }
    except Exception as e:
        raise ValueError(f"Failed to get coordinate info: {e}")

def convert_coordinates_to_indices(
    sim_dir: Union[str, Path],
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Convert coordinate ranges to index ranges for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        lon_range: Longitude range to convert
        lat_range: Latitude range to convert
        
    Returns:
        Dict: Index ranges and additional info
        
    Examples:
        >>> indices = convert_coordinates_to_indices(
        ...     "/path/to/sim", 
        ...     lon_range=(120.5, 121.5), 
        ...     lat_range=(23.5, 24.5)
        ... )
        >>> print(f"X indices: {indices['x_range']}")  
        >>> print(f"Y indices: {indices['y_range']}")
        >>> # Use these indices for precise selection
        >>> ds = load_indices("/path/to/sim", indices['x_range'], indices['y_range'])
    """
    from .coordinates.spatial_handler import (
        load_topo_dataset, extract_coordinates_from_topo,
        convert_coordinates_to_indices as _convert
    )
    
    try:
        topo_ds = load_topo_dataset(Path(sim_dir))
        coord_info = extract_coordinates_from_topo(topo_ds)
        topo_ds.close()
        
        result = _convert(coord_info, lon_range, lat_range)
        
        # Add coordinate information for reference
        if result['x_range'] is not None:
            x_min, x_max = result['x_range']
            result['actual_lon_range'] = (
                float(coord_info.lon[x_min]), 
                float(coord_info.lon[x_max])
            )
        
        if result['y_range'] is not None:
            y_min, y_max = result['y_range']
            result['actual_lat_range'] = (
                float(coord_info.lat[y_min]),
                float(coord_info.lat[y_max])
            )
            
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to convert coordinates: {e}")

def convert_indices_to_coordinates(
    sim_dir: Union[str, Path],
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None
) -> Dict:
    """
    Convert index ranges to coordinate ranges for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        x_range: X-index range to convert
        y_range: Y-index range to convert
        
    Returns:
        Dict: Coordinate ranges
        
    Examples:
        >>> coords = convert_indices_to_coordinates(
        ...     "/path/to/sim",
        ...     x_range=(100, 200),
        ...     y_range=(50, 150)
        ... )
        >>> print(f"Longitude range: {coords['lon_range']}")
        >>> print(f"Latitude range: {coords['lat_range']}")
    """
    from .coordinates.spatial_handler import (
        load_topo_dataset, extract_coordinates_from_topo,
        convert_indices_to_coordinates as _convert
    )
    
    try:
        topo_ds = load_topo_dataset(Path(sim_dir))
        coord_info = extract_coordinates_from_topo(topo_ds)
        topo_ds.close()
        
        return _convert(coord_info, x_range, y_range)
        
    except Exception as e:
        raise ValueError(f"Failed to convert indices: {e}")

def convert_time_to_indices(
    sim_dir: Union[str, Path],
    time_range: Tuple,
) -> Dict:
    """
    Convert a time range to the corresponding file index range using timestamps stored in
    the NetCDF outputs.
    
    Args:
        sim_dir: Simulation directory path
        time_range: Time range (start_time, end_time)

    Returns:
        Dict: Index range and additional info
    """
    from .coordinates.time_handler import normalize_time_range, read_time_from_file
    from .io.file_utils import parse_index_from_filename
    from .core.exceptions import validate_simulation_directory

    archive_dir = validate_simulation_directory(Path(sim_dir))
    nc_files = sorted(archive_dir.glob("*.nc"))

    if not nc_files:
        raise ValueError("No NetCDF files found in simulation archive")

    index_time_pairs = []
    for file_path in nc_files:
        try:
            index = parse_index_from_filename(file_path)
            timestamp = read_time_from_file(file_path)
        except Exception:
            continue
        index_time_pairs.append((index, timestamp))

    if not index_time_pairs:
        raise ValueError("Unable to extract timestamps from simulation files")

    # Ensure chronological ordering by timestamp
    index_time_pairs.sort(key=lambda item: item[1])

    tr_start, tr_end = normalize_time_range(time_range)

    if tr_start is None and tr_end is None:
        selected = index_time_pairs
    else:
        selected = [
            pair for pair in index_time_pairs
            if (tr_start is None or pair[1] >= tr_start)
            and (tr_end is None or pair[1] <= tr_end)
        ]

    if not selected:
        raise ValueError("No files found within the requested time range")

    selected.sort(key=lambda item: item[0])
    start_index = selected[0][0]
    end_index = selected[-1][0]

    return {
        'time_index_range': (start_index, end_index),
        'actual_start_time': selected[0][1],
        'actual_end_time': selected[-1][1],
        'timestamps': [ts for _, ts in selected],
    }

def convert_heights_to_indices(
    sim_dir: Union[str, Path],
    height_range: Tuple[float, float]
) -> Dict:
    """
    Convert height range to vertical index range for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        height_range: Height range (min_height, max_height) in meters
        
    Returns:
        Dict: Index range and additional info
        
    Examples:
        >>> indices = convert_heights_to_indices(
        ...     "/path/to/sim",
        ...     (0, 5000)  # 0 to 5km height
        ... )
        >>> print(f"Vertical indices: {indices['index_range']}")
        >>> # Use these indices for faster loading
        >>> vert_sel = VerticalSelection(index_range=indices['index_range'])
    """
    from .processing.vertical import read_vertical_levels_from_fort98
    
    try:
        levels = read_vertical_levels_from_fort98(Path(sim_dir))
        
        zmin, zmax = height_range
        z_min, z_max = min(float(zmin), float(zmax)), max(float(zmin), float(zmax))
        
        # Find indices within height range
        indices = np.where((levels >= z_min) & (levels <= z_max))[0]
        
        if indices.size == 0:
            raise ValueError(f"No vertical levels found in height range {height_range}")
        
        index_range = (int(indices.min()), int(indices.max()))
        
        return {
            'index_range': index_range,
            'actual_height_range': (float(levels[indices.min()]), float(levels[indices.max()])),
            'num_levels': len(indices),
            'all_heights': levels[indices].tolist()
        }
        
    except Exception as e:
        raise ValueError(f"Failed to convert heights to indices: {e}")

def create_variable_manifest(
    sim_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict:
    """
    Create a variable manifest for a simulation.
    
    Args:
        sim_dir: Simulation directory path
        output_path: Where to save the manifest
        **kwargs: Additional arguments for manifest creation
        
    Returns:
        Dict: Created variable manifest
    """
    from .io.manifest import create_manifest_for_simulation
    
    return create_manifest_for_simulation(
        sim_dir=Path(sim_dir),
        output_path=Path(output_path) if output_path else None,
        **kwargs
    )

# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Main API
    'open_vvm_dataset',
    
    # Convenience functions
    'quick_load',
    'load_surface_data', 
    'load_region',
    'load_indices',
    'load_time_series',
    
    # Utility functions
    'list_available_simulations',
    'get_simulation_info',
    'get_coordinate_info',
    'get_terrain_info',
    'convert_coordinates_to_indices',
    'convert_indices_to_coordinates',
    'convert_time_to_indices',
    'convert_heights_to_indices',
    'create_variable_manifest',
    
    # Type classes for structured API
    'Region',
    'TimeSelection', 
    'VerticalSelection',
    'ProcessingOptions',
    'LoadParameters'
]