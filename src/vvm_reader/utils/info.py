"""
VVM Reader Information Utilities

This module provides functions for querying simulation information including
simulation directories, coordinate systems, and terrain data.
"""

from pathlib import Path
from typing import Union, List, Dict, Optional
import numpy as np

from ..core.config import TOPO_VAR
from ..core.core_types import Region, CoordinateInfo, SliceInfo


# ============================================================================
# Simulation Discovery
# ============================================================================

def list_available_simulations(parent_dir: Union[str, Path]) -> List[Path]:
    """
    List all available VVM simulations in a directory.

    Args:
        parent_dir: Parent directory to search

    Returns:
        List[Path]: List of simulation directory paths
    """
    from ..io.file_utils import find_simulation_directories
    return find_simulation_directories(Path(parent_dir))


def get_simulation_info(sim_dir: Union[str, Path]) -> Dict:
    """
    Get information about a simulation directory.

    Args:
        sim_dir: Simulation directory path

    Returns:
        Dict: Simulation information
    """
    from ..io.file_utils import get_simulation_info as _get_info
    return _get_info(Path(sim_dir))


# ============================================================================
# Coordinate Information
# ============================================================================

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
    from ..coordinates.spatial import (
        load_topo_dataset, extract_coordinates_from_topo
    )

    try:
        topo_ds = load_topo_dataset(Path(sim_dir))
        coord_info = extract_coordinates_from_topo(topo_ds)
        topo_ds.close()

        return {
            'nx': len(coord_info.lon),
            'ny': len(coord_info.lat),
            'lon_range': (float(coord_info.lon.min()), float(coord_info.lon.max())),
            'lat_range': (float(coord_info.lat.min()), float(coord_info.lat.max())),
            'x_dim': coord_info.x_dim,
            'y_dim': coord_info.y_dim,
        }
    except Exception as e:
        raise ValueError(f"Failed to get coordinate info: {e}")


# ============================================================================
# Terrain Information
# ============================================================================

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
        from ..coordinates.spatial import (
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


# ============================================================================
# Vertical Information
# ============================================================================

def get_vertical_info(sim_dir: Union[str, Path]) -> Dict:
    """
    Get information about vertical coordinate system.

    Args:
        sim_dir: Simulation directory path

    Returns:
        Dict: Vertical coordinate information including levels, heights, and spacing

    Examples:
        >>> vert_info = get_vertical_info("/path/to/sim")
        >>> if vert_info['has_fort98']:
        ...     print(f"Vertical levels: {vert_info['num_levels']}")
        ...     print(f"Height range: {vert_info['height_range']}")
        ...     print(f"Spacing: {vert_info['level_spacing']}")
    """
    from ..processing.vertical import read_vertical_levels_from_fort98
    from ..core.config import FORT98_FILENAME
    from ..core.exceptions import RequiredFileNotFoundError, DataProcessingError

    sim_path = Path(sim_dir)
    info = {
        'has_fort98': False,
        'num_levels': None,
        'height_range': None,
        'level_spacing': None,
        'fort98_path': str(sim_path / FORT98_FILENAME),
        'error': None
    }

    try:
        levels = read_vertical_levels_from_fort98(sim_path)
        info['has_fort98'] = True
        info['num_levels'] = len(levels)
        info['height_range'] = (float(levels.min()), float(levels.max()))

        if len(levels) > 1:
            spacing = np.diff(levels)
            info['level_spacing'] = {
                'mean': float(np.mean(spacing)),
                'min': float(np.min(spacing)),
                'max': float(np.max(spacing)),
                'uniform': bool(np.allclose(spacing, spacing[0], rtol=1e-6))
            }
    except RequiredFileNotFoundError:
        info['error'] = 'fort.98 file not found'
    except DataProcessingError as e:
        info['error'] = f'fort.98 processing error: {e.reason}'
    except Exception as e:
        info['error'] = f'Unexpected error: {e}'

    return info


# ============================================================================
# Spatial Information
# ============================================================================

def get_spatial_info(coord_info: CoordinateInfo, slice_info: SliceInfo) -> Dict:
    """
    Get spatial domain information.

    Args:
        coord_info: Coordinate information
        slice_info: Slice information

    Returns:
        Dict: Spatial domain information including grid sizes and ranges

    Examples:
        >>> from vvm_reader.coordinates.spatial import load_topo_dataset, extract_coordinates_from_topo, compute_regional_slices
        >>> from vvm_reader.core.core_types import Region
        >>>
        >>> topo_ds = load_topo_dataset(sim_dir)
        >>> coord_info = extract_coordinates_from_topo(topo_ds)
        >>> region = Region(x_range=(100, 200), y_range=(50, 150))
        >>> slice_info = compute_regional_slices(coord_info, region)
        >>> spatial_info = get_spatial_info(coord_info, slice_info)
        >>> print(f"Selected: {spatial_info['nx_selected']} x {spatial_info['ny_selected']}")
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
