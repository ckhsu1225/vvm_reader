"""
VVM Reader Main Interface

This module provides the main API functions for loading VVM datasets.
Utility functions have been moved to the utils package.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple
import xarray as xr

from .core.core_types import (
    LoadParameters, Region, TimeSelection, VerticalSelection, ProcessingOptions
)
from .io.dataset_loader import load_vvm_dataset

# Get logger for this module
logger = logging.getLogger('vvm_reader.main')

# Import utility functions for backward compatibility
from .utils import (
    list_available_simulations,
    get_simulation_info,
    get_coordinate_info,
    get_terrain_info,
    get_vertical_info,
    get_spatial_info,
    convert_coordinates_to_indices,
    convert_indices_to_coordinates,
    convert_time_to_indices,
    convert_heights_to_indices,
    create_variable_manifest,
)


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
    var_manifest: Optional[Union[str, Path, dict]] = None,
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
            logger.warning(
                "Could not auto-detect vertical range (%s). Loading all levels.",
                e
            )

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

    Note: This function is a convenience wrapper around load_region().
    Consider using load_region() directly for more flexibility.

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

        # Equivalent using load_region():
        >>> ds = load_region("/path/to/sim", x_range=(50, 149), y_range=(50, 149),
        ...                  variables=["u", "v", "w"])
    """
    # Delegate to load_region() for better maintainability
    return load_region(
        sim_dir=sim_dir,
        x_range=x_range,
        y_range=y_range,
        variables=variables,
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

    # Utility functions (re-exported from utils for backward compatibility)
    'list_available_simulations',
    'get_simulation_info',
    'get_coordinate_info',
    'get_terrain_info',
    'get_vertical_info',
    'get_spatial_info',
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
