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
    auto_compute_diagnostics: bool = True,
) -> xr.Dataset:
    """
    Load VVM dataset with structured parameters.

    This is the main interface for loading VVM data using structured parameter objects
    for better organization and type safety. Automatically computes diagnostic variables
    when requested in the variables list.

    Args:
        sim_dir: Path to simulation directory
        groups: Output groups to load (e.g., ["L.Dynamic", "L.Radiation"])
        variables: Specific variables to load (can include both file and diagnostic variables)
        region: Spatial region selection
        time_selection: Time selection parameters
        vertical_selection: Vertical level selection parameters
        processing_options: Data processing options
        var_manifest: Variable manifest (path, dict, or None for auto-load)
        auto_compute_diagnostics: Automatically compute diagnostic variables (default: True)
            Set to False to disable automatic computation and compute manually later.

    Returns:
        xr.Dataset: Loaded and processed VVM dataset

    Examples:
        # Load file and diagnostic variables together (automatic computation)
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["th", "qv", "t", "rh", "hm"]  # t, rh, hm auto-computed
        ... )

        # Load specific variables with spatial/temporal selection
        >>> region = Region(lon_range=(120, 122), lat_range=(23, 25))
        >>> time_sel = TimeSelection(time_index_range=(0, 36))
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["u", "v", "w", "th"],
        ...     region=region,
        ...     time_selection=time_sel
        ... )

        # Disable automatic diagnostic computation (manual control)
        >>> ds = open_vvm_dataset(
        ...     "/path/to/sim",
        ...     variables=["th", "qv"],
        ...     auto_compute_diagnostics=False
        ... )
        >>> ds = vvm.compute_diagnostics(ds, ["t", "rh"], sim_dir)
    """
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
        region=region or Region(),
        time_selection=time_selection or TimeSelection(),
        vertical_selection=vertical_selection or VerticalSelection(),
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
