"""
VVM Reader Terrain Processing

This module handles terrain masking and related topography operations.
"""

import numpy as np
import xarray as xr

from ..core.config import VERTICAL_DIM
from ..core.exceptions import DataProcessingError
from ..core.core_types import CoordinateInfo, SliceInfo

# ============================================================================
# Terrain Masking
# ============================================================================

def apply_terrain_mask(
    dataset: xr.Dataset,
    topo: xr.DataArray,
    variables: list = None,
    mask_value: float = np.nan,
    vertical_dim: str = VERTICAL_DIM,
    vertical_offset: int = 0,
    vertical_indices: np.ndarray = None
) -> xr.Dataset:
    """
    Apply terrain masking to 3D variables.

    Sets values to specified mask_value for grid points inside or at the terrain surface.
    Masking rule: k <= topo (where k is 1-based vertical index).

    Performance note: This function uses dynamic broadcasting to create the mask
    only when needed, which is much faster and uses less memory than loading a
    pre-computed 3D mask.

    Args:
        dataset: Input dataset
        topo: 2D topography data (y, x) with terrain top indices
        variables: List of variable names to mask. If None, mask all 3D variables.
        mask_value: Value to use for masking (default: np.nan, use 0.0 for winds before centering)
        vertical_dim: Name of vertical dimension
        vertical_offset: Offset for vertical indices when using contiguous vertical slicing
        vertical_indices: Explicit array of original k indices (1-based) for arbitrary selection.
                         Takes priority over vertical_offset when provided.

    Returns:
        xr.Dataset: Dataset with terrain-masked variables

    Raises:
        DataProcessingError: If masking operation fails
    """
    if vertical_dim not in dataset.sizes:
        return dataset

    try:
        # Get vertical dimension info
        nz = dataset.sizes[vertical_dim]
        y_dim, x_dim = topo.dims

        # Create 1-based vertical index array
        if vertical_indices is not None:
            # Arbitrary selection: use provided k indices (convert to 1-based if needed)
            # The indices should already be the original k values
            k_indices = xr.DataArray(
                (vertical_indices + 1).astype(np.int32),  # Convert 0-based to 1-based
                dims=[vertical_dim]
            )
        else:
            # Contiguous selection: use offset-based calculation
            k_indices = xr.DataArray(
                np.arange(1 + vertical_offset, nz + 1 + vertical_offset, dtype=np.int32),
                dims=[vertical_dim]
            )

        # Convert topo to integer for comparison
        topo_int = topo.astype(np.int32)

        # Create mask: True where k <= topo (inside terrain)
        # Broadcasting creates mask over (vertical_dim, y_dim, x_dim)
        terrain_mask = k_indices <= topo_int

        # Apply mask to specified or all 3D variables
        masked_vars = {}
        for name, var in dataset.data_vars.items():
            if (vertical_dim in var.sizes and
                y_dim in var.sizes and
                x_dim in var.sizes):
                # Only mask if variable is in the list (or list is None)
                if variables is None or name in variables:
                    # Mask terrain points to specified value
                    # Use .data to avoid xarray overhead
                    masked_var = var.where(~terrain_mask.data, mask_value)
                    masked_vars[name] = masked_var
                else:
                    masked_vars[name] = var
            else:
                # Keep 2D and other variables unchanged
                masked_vars[name] = var

        return xr.Dataset(
            data_vars=masked_vars,
            coords=dataset.coords,
            attrs=dataset.attrs
        )

    except Exception as e:
        raise DataProcessingError("terrain masking", str(e))


def center_staggered_variables(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo,
    topo: xr.DataArray,
    center_suffix: str = "_c"
) -> tuple:
    """
    Center staggered variables (winds and vorticities) to cell centers with terrain awareness.

    Centering rules for single-direction stagger:
    - u: avg of u(i) and u(i-1), periodic only for full domain
    - v: avg of v(j) and v(j-1), periodic only for full domain
    - w: avg of w(k) and w(k-1), non-periodic (bottom ghost = NaN)

    Centering rules for dual-direction stagger:
    - zeta (x,y): avg of 4 corner points
    - eta (x,z): avg of 4 corner points
    - xi (y,z): avg of 4 corner points

    Args:
        dataset: Input dataset
        coord_info: Coordinate information
        slice_info: Slice information for periodicity
        topo: 2D topography data with terrain top indices
        center_suffix: Suffix for centered variables

    Returns:
        tuple: (updated_dataset, list_of_new_variable_names)
    """
    from ..core.config import STAGGERED_VARIABLES, STAGGER_CONFIG

    vertical_dim = VERTICAL_DIM
    x_dim, y_dim = coord_info.x_dim, coord_info.y_dim

    # Check which staggered variables are present
    available_vars = [var for var in STAGGERED_VARIABLES if var in dataset.data_vars]
    if not available_vars:
        return dataset, []

    new_variables = {}
    new_var_names = []

    # Process each variable based on its stagger configuration
    for var_name in available_vars:
        if var_name not in dataset.data_vars:
            continue

        var_data = dataset[var_name]
        stagger_dims = STAGGER_CONFIG.get(var_name, ())

        if not stagger_dims:
            continue

        # Center the variable based on stagger configuration
        if len(stagger_dims) == 1:
            # Single-direction stagger (u, v, w)
            centered = _center_single_direction(
                var_data, stagger_dims[0],
                x_dim, y_dim, vertical_dim,
                slice_info
            )
        elif len(stagger_dims) == 2:
            # Dual-direction stagger (zeta, eta, xi)
            centered = _center_dual_direction(
                var_data, stagger_dims,
                x_dim, y_dim, vertical_dim,
                slice_info
            )
        else:
            continue

        if centered is not None:
            # Preserve attributes and encoding
            centered.attrs.update(var_data.attrs)
            centered.encoding = getattr(var_data, 'encoding', {})

            # Add to new variables
            new_name = f"{var_name}{center_suffix}" if center_suffix else var_name
            new_variables[new_name] = centered
            new_var_names.append(new_name)

    if new_variables:
        dataset = dataset.assign(**new_variables)

    return dataset, new_var_names


def _safe_shift(var: xr.DataArray, shifts: dict) -> xr.DataArray:
    """
    Safely shift variable, handling dask edge cases where dimension size is small.
    
    When using dask, shifting a dimension with size <= shift amount can cause
    ZeroDivisionError or other issues during chunk calculation. This function
    checks for this condition and returns an all-NaN array instead.
    """
    # Check if any shift operation would crash dask (size <= shift)
    for dim, shift in shifts.items():
        if dim in var.sizes and var.sizes[dim] <= abs(shift):
             return xr.full_like(var, np.nan)
    
    return var.shift(shifts).fillna(np.nan)


def _center_single_direction(
    var: xr.DataArray,
    stagger_dim: str,
    x_dim: str,
    y_dim: str,
    z_dim: str,
    slice_info: SliceInfo,
) -> xr.DataArray:
    """
    Center a variable staggered in a single direction.

    Args:
        var: Variable to center
        stagger_dim: Dimension where variable is staggered ('x', 'y', or 'z')
        x_dim, y_dim, z_dim: Actual dimension names in dataset
        slice_info: Slice information for periodicity

    Returns:
        Centered variable
    """
    if stagger_dim == 'x':
        actual_dim = x_dim
        is_periodic = slice_info.periodic_x
    elif stagger_dim == 'y':
        actual_dim = y_dim
        is_periodic = slice_info.periodic_y
    elif stagger_dim == 'z':
        actual_dim = z_dim
        is_periodic = False  # Vertical is never periodic
    else:
        return None

    if actual_dim not in var.sizes:
        return None

    # Perform centering
    if is_periodic:
        # Periodic: use roll
        var_shifted = var.roll({actual_dim: 1}, roll_coords=False)
    else:
        # Non-periodic: use safe shift
        var_shifted = _safe_shift(var, {actual_dim: 1})

    # Average the 2 points (xarray handles NaN properly in mean)
    # Stack the 2 points along a temporary dimension and take mean
    stacked = xr.concat([var, var_shifted], dim='_temp_avg')
    centered = stacked.mean(dim='_temp_avg')
    return centered


def _center_dual_direction(
    var: xr.DataArray,
    stagger_dims: tuple,
    x_dim: str,
    y_dim: str,
    z_dim: str,
    slice_info: SliceInfo
) -> xr.DataArray:
    """
    Center a variable staggered in two directions (e.g., zeta, eta, xi).

    Uses 4-point averaging from corner points to cell center.

    Args:
        var: Variable to center
        stagger_dims: Tuple of dimensions where variable is staggered (e.g., ('x', 'y'))
        x_dim, y_dim, z_dim: Actual dimension names in dataset
        slice_info: Slice information for periodicity

    Returns:
        Centered variable
    """
    # Map stagger dimension names to actual dimension names and periodicity
    dim_map = {
        'x': (x_dim, slice_info.periodic_x),
        'y': (y_dim, slice_info.periodic_y),
        'z': (z_dim, False)  # Vertical never periodic
    }

    actual_dims = []
    periodicities = []
    for stagger_dim in stagger_dims:
        if stagger_dim in dim_map:
            actual_dim, is_periodic = dim_map[stagger_dim]
            if actual_dim in var.sizes:
                actual_dims.append(actual_dim)
                periodicities.append(is_periodic)

    if len(actual_dims) != 2:
        return None

    dim1, dim2 = actual_dims
    periodic1, periodic2 = periodicities

    # Get 4 corner points for averaging
    # Point 1: original position (i, j)
    p1 = var

    # Point 2: shifted in dim1 (i-1, j)
    if periodic1:
        p2 = var.roll({dim1: 1}, roll_coords=False)
    else:
        p2 = _safe_shift(var, {dim1: 1})

    # Point 3: shifted in dim2 (i, j-1)
    if periodic2:
        p3 = var.roll({dim2: 1}, roll_coords=False)
    else:
        p3 = _safe_shift(var, {dim2: 1})

    # Point 4: shifted in both dims (i-1, j-1)
    if periodic1 and periodic2:
        p4 = var.roll({dim1: 1, dim2: 1}, roll_coords=False)
    elif periodic1 and not periodic2:
        p4 = _safe_shift(var.roll({dim1: 1}, roll_coords=False), {dim2: 1})
    elif not periodic1 and periodic2:
        p4 = _safe_shift(var.roll({dim2: 1}, roll_coords=False), {dim1: 1})
    else:
        p4 = _safe_shift(var, {dim1: 1, dim2: 1})

    # Average the 4 points (xarray handles NaN properly in mean)
    # Stack the 4 points along a temporary dimension and take mean
    stacked = xr.concat([p1, p2, p3, p4], dim='_temp_avg')
    centered = stacked.mean(dim='_temp_avg')

    return centered
