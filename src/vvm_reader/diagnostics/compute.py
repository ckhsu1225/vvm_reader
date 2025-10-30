"""
Diagnostic Computation Engine

This module provides the main computation engine for diagnostic variables,
including dependency resolution and automatic calculation.
"""

import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional, Set

from .registry import get_registry
from ..utils.info import get_reference_profiles
from ..core.exceptions import DataProcessingError

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# Main Computation Function
# ============================================================================

def compute_diagnostics(
    ds: xr.Dataset,
    variables: List[str],
    sim_dir: Path,
    profiles: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Compute diagnostic variables and add them to the dataset.

    Args:
        ds: Input dataset containing model output variables
        variables: List of diagnostic variable names to compute
        sim_dir: Simulation directory (needed for fort.98 reading)
        profiles: Optional pre-loaded reference profiles (xr.Dataset).
                  If None, will read from fort.98 using get_reference_profiles()

    Returns:
        Dataset with diagnostic variables added

    Raises:
        KeyError: If a requested variable is not registered
        DataProcessingError: If computation fails

    Example:
        >>> ds = vvm.open_vvm_dataset("/path/to/sim", variables=["th", "qv"])
        >>> ds = compute_diagnostics(ds, ["T", "RH"], sim_dir="/path/to/sim")
    """
    registry = get_registry()

    # Check if all variables are registered
    for var in variables:
        if not registry.is_registered(var):
            available = registry.list_all()
            raise KeyError(
                f"Diagnostic variable '{var}' not registered. "
                f"Available: {available}"
            )

    # Load reference profiles if not provided
    if profiles is None:
        try:
            # Use get_reference_profiles to get xr.Dataset with lev coordinates
            profiles = get_reference_profiles(Path(sim_dir))
            logger.debug(f"Loaded reference profiles from {sim_dir}/fort.98")
        except Exception as e:
            logger.warning(f"Could not load fort.98 profiles: {e}")
            profiles = xr.Dataset()

    # Align profiles with dataset vertical dimension if needed
    if isinstance(profiles, xr.Dataset) and 'lev' in profiles.dims:
        if 'lev' in ds.dims:
            # Case 1: Dataset has vertical dimension - align profiles to it
            try:
                # Use nearest neighbor matching with tolerance to handle floating point errors
                profiles_aligned = profiles.sel(lev=ds['lev'], method='nearest', tolerance=1.0)

                # Replace the lev coordinate with dataset's lev to ensure exact match
                # This prevents NaN values due to floating point differences
                profiles_aligned = profiles_aligned.assign_coords({'lev': ds['lev']})

                logger.debug(
                    f"Aligned profiles to dataset lev dimension: "
                    f"{len(ds['lev'])} levels from {ds['lev'].values[0]:.1f}m to {ds['lev'].values[-1]:.1f}m"
                )
                profiles = profiles_aligned
            except Exception as e:
                logger.warning(f"Could not align profiles with dataset levels: {e}")
                # Continue with full profiles - xarray will handle alignment during operations

        elif 'surface_level_index' in ds.data_vars:
            # Case 2: Surface-only dataset with surface_level_index
            # Select profile values at the surface level for each grid point
            try:
                sfc_indices = ds['surface_level_index'].astype(int)

                # Extract profile values at surface indices for each profile variable
                profiles_surface = xr.Dataset()
                for var_name in profiles.data_vars:
                    profile_var = profiles[var_name]
                    # Select values at surface indices (varying by horizontal position)
                    surface_values = profile_var.isel(lev=sfc_indices)
                    profiles_surface[var_name] = surface_values

                logger.debug(
                    f"Selected surface profile values using surface_level_index "
                    f"(k range: {int(sfc_indices.min())}-{int(sfc_indices.max())})"
                )
                profiles = profiles_surface

            except Exception as e:
                logger.warning(f"Could not select surface profiles: {e}")
                # Fall back to using the lowest level (k=0) for all grid points
                try:
                    profiles_surface = profiles.isel(lev=0)
                    logger.debug("Using lowest level (k=0) profile values for all grid points")
                    profiles = profiles_surface
                except Exception as e2:
                    logger.warning(f"Could not select k=0 profiles: {e2}")
                    # Continue with full profiles

    # Resolve computation order
    try:
        computation_order = registry.resolve_computation_order(variables)
        logger.debug(f"Computation order: {computation_order}")
    except ValueError as e:
        raise DataProcessingError("dependency resolution", str(e))

    # Compute diagnostics in order
    diagnostics = {}

    for var_name in computation_order:
        logger.debug(f"Computing diagnostic variable: {var_name}")

        diag_var = registry.get(var_name)
        if diag_var is None:
            raise DataProcessingError(
                "diagnostic computation",
                f"Variable '{var_name}' not found in registry"
            )

        # Check if required file dependencies are available
        missing_deps = diag_var.file_dependencies - set(ds.data_vars)
        if missing_deps:
            raise DataProcessingError(
                "diagnostic computation",
                f"Missing required variables for '{var_name}': {missing_deps}"
            )

        available_profiles = set(profiles.data_vars)
        missing_profiles = diag_var.profile_dependencies - available_profiles
        if missing_profiles:
            raise DataProcessingError(
                "diagnostic computation",
                f"Missing required profiles for '{var_name}': {missing_profiles}"
            )

        # Compute the variable
        try:
            result = diag_var.compute_func(ds, profiles, diagnostics)

            if result is not None:
                diagnostics[var_name] = result
                logger.debug(f"Successfully computed '{var_name}'")
            else:
                logger.warning(f"Computation of '{var_name}' returned None")

        except Exception as e:
            raise DataProcessingError(
                f"computing '{var_name}'",
                f"{type(e).__name__}: {e}"
            )

    # Add computed diagnostics to dataset
    ds_out = ds.copy()
    for var_name, var_data in diagnostics.items():
        if var_name in variables:  # Only add requested variables
            ds_out[var_name] = var_data

    return ds_out


def separate_file_and_diagnostic_variables(
    variables: List[str]
) -> tuple[Set[str], Set[str]]:
    """
    Separate a list of variables into file variables and diagnostic variables.

    Args:
        variables: List of variable names (mix of file and diagnostic variables)

    Returns:
        Tuple of (file_variables, diagnostic_variables)

    Example:
        >>> file_vars, diag_vars = separate_file_and_diagnostic_variables(
        ...     ["u", "v", "t", "rh"]
        ... )
        >>> print(file_vars)  # {'u', 'v'}
        >>> print(diag_vars)  # {'t', 'rh'}
    """
    registry = get_registry()

    file_vars = set()
    diag_vars = set()

    for var in variables:
        if registry.is_registered(var):
            diag_vars.add(var)
        else:
            file_vars.add(var)

    return file_vars, diag_vars


def get_required_file_variables(diagnostic_variables: List[str]) -> Set[str]:
    """
    Get all file variables required to compute diagnostic variables.

    This performs recursive dependency resolution to find all model output
    variables needed.

    Args:
        diagnostic_variables: List of diagnostic variable names

    Returns:
        Set of file variable names needed

    Example:
        >>> required = get_required_file_variables(["t", "rh"])
        >>> print(required)  # {'th', 'qv'}
    """
    registry = get_registry()

    try:
        file_deps = registry.get_file_dependencies(diagnostic_variables)
        logger.debug(f"Required file variables for {diagnostic_variables}: {file_deps}")
        return file_deps
    except KeyError as e:
        raise KeyError(f"Unknown diagnostic variable: {e}")


def get_required_profiles(diagnostic_variables: List[str]) -> Set[str]:
    """
    Get all fort.98 profiles required to compute diagnostic variables.

    Args:
        diagnostic_variables: List of diagnostic variable names

    Returns:
        Set of profile names needed (e.g., 'PIBAR', 'RHO')

    Example:
        >>> profiles = get_required_profiles(["t", "rh"])
        >>> print(profiles)  # {'PIBAR', 'PBAR'}
    """
    registry = get_registry()

    try:
        profile_deps = registry.get_profile_dependencies(diagnostic_variables)
        logger.debug(f"Required profiles for {diagnostic_variables}: {profile_deps}")
        return profile_deps
    except KeyError as e:
        raise KeyError(f"Unknown diagnostic variable: {e}")


def list_available_diagnostics() -> List[str]:
    """
    List all available diagnostic variables.

    Returns:
        Sorted list of diagnostic variable names

    Example:
        >>> diagnostics = list_available_diagnostics()
        >>> print(diagnostics)
        ['cwv', 'iwp', 'lwp', 'sd', 'hm', 'hms',
         't', 'tv', 'qvs', 'rh', 'the', 'thes', 'thv', 'ws']
    """
    registry = get_registry()
    return registry.list_all()


def get_diagnostic_metadata(variable: str) -> Dict[str, str]:
    """
    Get metadata for a diagnostic variable.

    Args:
        variable: Diagnostic variable name

    Returns:
        Dictionary with metadata (long_name, units, description, etc.)

    Raises:
        KeyError: If variable is not registered

    Example:
        >>> metadata = get_diagnostic_metadata('t')
        >>> print(metadata['long_name'])  # 'Temperature'
        >>> print(metadata['units'])      # 'K'
    """
    registry = get_registry()
    return registry.get_metadata(variable)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'compute_diagnostics',
    'separate_file_and_diagnostic_variables',
    'get_required_file_variables',
    'get_required_profiles',
    'list_available_diagnostics',
    'get_diagnostic_metadata',
]
