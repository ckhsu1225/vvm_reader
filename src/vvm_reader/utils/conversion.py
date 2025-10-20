"""
VVM Reader Conversion Utilities

This module provides functions for converting between different coordinate
systems and representations (coordinates ↔ indices, time ↔ indices, heights ↔ indices).
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict
import numpy as np

from ..coordinates.time_handler import normalize_time_range, read_time_from_file
from ..io.file_utils import parse_index_from_filename
from ..core.exceptions import validate_simulation_directory


# ============================================================================
# Spatial Coordinate Conversion
# ============================================================================

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
        >>> ds = load_region("/path/to/sim", x_range=indices['x_range'], y_range=indices['y_range'])
    """
    from ..coordinates.spatial import (
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
    from ..coordinates.spatial import (
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


# ============================================================================
# Time Conversion
# ============================================================================

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


# ============================================================================
# Vertical Conversion
# ============================================================================

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
    from ..processing.vertical import read_vertical_levels_from_fort98

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


# ============================================================================
# Manifest Creation
# ============================================================================

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
    from ..io.manifest import create_manifest_for_simulation

    return create_manifest_for_simulation(
        sim_dir=Path(sim_dir),
        output_path=Path(output_path) if output_path else None,
        **kwargs
    )
