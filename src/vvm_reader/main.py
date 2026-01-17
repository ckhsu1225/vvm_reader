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
    get_terrain_height,
    get_reference_profiles,
    get_coriolis_info,
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
    # Structured parameter objects (for advanced usage)
    region: Optional[Region] = None,
    time_selection: Optional[TimeSelection] = None,
    vertical_selection: Optional[VerticalSelection] = None,
    processing_options: Optional[ProcessingOptions] = None,
    # Flat parameters for Region (simpler API)
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
    # Flat parameters for TimeSelection (simpler API)
    time_range: Optional[Tuple] = None,
    time_index_range: Optional[Tuple[int, int]] = None,
    time_indices: Optional[Sequence[int]] = None,
    time_values: Optional[Sequence] = None,
    # Flat parameters for VerticalSelection (simpler API)
    vertical_index_range: Optional[Tuple[int, int]] = None,
    height_range: Optional[Tuple[float, float]] = None,
    heights: Optional[Sequence[float]] = None,
    level_indices: Optional[Sequence[int]] = None,
    surface_nearest: Optional[bool] = None,
    surface_only: Optional[bool] = None,
    # Other options
    var_manifest: Optional[Union[str, Path, dict]] = None,
    auto_compute_diagnostics: bool = True,
) -> xr.Dataset:
    """
    Load VVM dataset with flexible parameter options.

    This is the main interface for loading VVM data. Supports both:
    1. Simple flat parameters (recommended for most use cases)
    2. Structured parameter objects (for advanced usage)

    When both flat parameters and object parameters are provided, flat parameters
    take priority and override the corresponding values in the objects.

    Args:
        sim_dir: Path to simulation directory
        groups: Output groups to load (e.g., ["L.Dynamic", "L.Radiation"])
        variables: Specific variables to load (can include both file and diagnostic variables)

        # Structured parameter objects (advanced)
        region: Spatial region selection object
        time_selection: Time selection parameters object
        vertical_selection: Vertical level selection parameters object
        processing_options: Data processing options object

        # Flat parameters for spatial selection (simple)
        lon_range: Longitude range (min_lon, max_lon) in degrees
        lat_range: Latitude range (min_lat, max_lat) in degrees
        x_range: X-index range (min_x, max_x) - takes priority over lon_range
        y_range: Y-index range (min_y, max_y) - takes priority over lat_range

        # Flat parameters for time selection (simple)
        time_range: Time range (start_time, end_time) as datetime objects
        time_index_range: File index range (start_index, end_index)
        time_indices: Arbitrary list of file indices (e.g., [0, 5, 10, 20])
        time_values: Arbitrary list of time values

        # Flat parameters for vertical selection (simple)
        vertical_index_range: Vertical index range (start_level, end_level) - contiguous
        height_range: Height range (min_height, max_height) in meters - contiguous
        heights: Arbitrary list of heights in meters (e.g., [500, 1000, 3000])
        level_indices: Arbitrary list of level indices (e.g., [0, 5, 10, 20])
        surface_nearest: Extract surface-nearest values
        surface_only: Keep only surface values (removes vertical dimension)

        # Other options
        var_manifest: Variable manifest (path, dict, or None for auto-load)
        auto_compute_diagnostics: Automatically compute diagnostic variables (default: True)

    Returns:
        xr.Dataset: Loaded and processed VVM dataset

    Examples:
        # Simple usage with flat parameters (recommended)
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["th", "qv", "t"],
        ...     lon_range=(120, 122),
        ...     lat_range=(23, 25),
        ...     time_index_range=(0, 36),
        ...     height_range=(0, 5000)
        ... )

        # Load at arbitrary heights (uses nearest levels)
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["th", "w"],
        ...     heights=[500, 1000, 3000, 5000]
        ... )

        # Load with index-based selection
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["u", "v", "w"],
        ...     x_range=(100, 200),
        ...     y_range=(50, 150),
        ...     time_index_range=(0, 10)
        ... )

        # Advanced usage with structured objects
        >>> region = Region(lon_range=(120, 122), lat_range=(23, 25))
        >>> time_sel = TimeSelection(time_index_range=(0, 36))
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["u", "v", "w", "th"],
        ...     region=region,
        ...     time_selection=time_sel
        ... )

        # Mixed: use object as base, override with flat params
        >>> region = Region(lon_range=(120, 125), lat_range=(22, 26))
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     region=region,
        ...     lon_range=(121, 122)  # Override lon_range only
        ... )
    """
    # Merge flat parameters with object parameters
    # Flat parameters take priority over object parameters

    # Build Region
    effective_region = region or Region()
    if lon_range is not None or lat_range is not None or x_range is not None or y_range is not None:
        effective_region = Region(
            lon_range=lon_range if lon_range is not None else effective_region.lon_range,
            lat_range=lat_range if lat_range is not None else effective_region.lat_range,
            x_range=x_range if x_range is not None else effective_region.x_range,
            y_range=y_range if y_range is not None else effective_region.y_range,
        )

    # Build TimeSelection
    effective_time = time_selection or TimeSelection()
    if time_range is not None or time_index_range is not None or time_indices is not None or time_values is not None:
        effective_time = TimeSelection(
            time_range=time_range if time_range is not None else effective_time.time_range,
            time_index_range=time_index_range if time_index_range is not None else effective_time.time_index_range,
            time_indices=time_indices if time_indices is not None else effective_time.time_indices,
            time_values=time_values if time_values is not None else effective_time.time_values,
        )

    # Build VerticalSelection
    effective_vertical = vertical_selection or VerticalSelection()
    has_vertical_flat = any([
        vertical_index_range is not None,
        height_range is not None,
        heights is not None,
        level_indices is not None,
        surface_nearest is not None,
        surface_only is not None
    ])
    if has_vertical_flat:
        effective_vertical = VerticalSelection(
            index_range=vertical_index_range if vertical_index_range is not None else effective_vertical.index_range,
            height_range=height_range if height_range is not None else effective_vertical.height_range,
            heights=heights if heights is not None else effective_vertical.heights,
            level_indices=level_indices if level_indices is not None else effective_vertical.level_indices,
            surface_nearest=surface_nearest if surface_nearest is not None else effective_vertical.surface_nearest,
            surface_only=surface_only if surface_only is not None else effective_vertical.surface_only,
        )
    sim_path = Path(sim_dir)

    # Separate file variables and diagnostic variables
    file_vars = variables
    diag_vars = []

    if variables and auto_compute_diagnostics:
        try:
            from .diagnostics import separate_file_and_diagnostic_variables, get_required_file_variables

            file_vars, diag_vars = separate_file_and_diagnostic_variables(variables)

            if diag_vars:
                # Get file variables required by diagnostic variables
                required_file_vars = get_required_file_variables(diag_vars)

                # Combine user-requested file vars with required file vars
                all_file_vars = list(file_vars | required_file_vars)

                logger.info(
                    "Diagnostic variables requested: %s", diag_vars
                )
                logger.debug(
                    "Auto-loading required file variables: %s", required_file_vars
                )

                file_vars = all_file_vars
        except ImportError:
            logger.warning("Diagnostics module not available, skipping diagnostic computation")
            diag_vars = []
        except Exception as e:
            logger.warning(
                "Failed to separate diagnostic variables: %s. "
                "Set auto_compute_diagnostics=False to disable.",
                e
            )
            diag_vars = []

    # Create parameter object with file variables only
    params = LoadParameters(
        groups=groups,
        variables=file_vars,
        region=effective_region,
        time_selection=effective_time,
        vertical_selection=effective_vertical,
        processing_options=processing_options or ProcessingOptions(),
        var_manifest=var_manifest
    )

    # Load file variables
    ds = load_vvm_dataset(sim_path, params)

    # Compute diagnostic variables if requested
    if diag_vars and auto_compute_diagnostics:
        try:
            from .diagnostics import compute_diagnostics

            logger.info("Computing diagnostic variables: %s", diag_vars)
            ds = compute_diagnostics(ds, diag_vars, sim_path)

        except Exception as e:
            logger.error(
                "Failed to compute diagnostic variables %s: %s",
                diag_vars, e
            )
            logger.info(
                "Suggestion: Set auto_compute_diagnostics=False and compute manually using "
                "vvm.compute_diagnostics(ds, %s, sim_dir)",
                diag_vars
            )
            raise

    return ds


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
    'get_terrain_height',
    'get_reference_profiles',
    'get_coriolis_info',
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
