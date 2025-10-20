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
    terrain_mask: xr.DataArray
) -> xr.Dataset:
    """
    Apply terrain masking to 3D variables.
    
    Args:
        dataset: Input dataset
        terrain_mask: 3D terrain mask array
    
    Returns:
        xr.Dataset: Masked dataset
    """
    try:
        z_dim, y_dim, x_dim = terrain_mask.sizes
        mask_bool = terrain_mask.astype(bool)

        masked_vars = {}
        for name, var in dataset.data_vars.items():
            if (z_dim in var.sizes and
                y_dim in var.sizes and
                x_dim in var.sizes):
                masked_var = var.where(mask_bool.values)
                masked_vars[name] = masked_var
            else:
                masked_vars[name] = var

        return xr.Dataset(
            data_vars=masked_vars,
            coords=dataset.coords,
            attrs=dataset.attrs
        )

    except Exception as e:
        raise DataProcessingError("terrain masking", str(e))

def mask_staggered_variables_for_centering(
    dataset: xr.Dataset,
    variables: list,
    terrain_mask: xr.DataArray,
    mask_value: float = np.nan
) -> xr.Dataset:
    """
    Mask staggered variables inside terrain before centering operations.

    Args:
        dataset: Input dataset
        variables: List of variable names to mask
        terrain_mask: 3D terrain mask array
        mask_value: Value to use for masking (0.0 for winds, np.nan for vorticity)

    Returns:
        xr.Dataset: Dataset with variables masked inside terrain
    """
    try:
        z_dim, y_dim, x_dim = terrain_mask.sizes
        mask_bool = terrain_mask.astype(bool)

        updates = {}
        for name, var in dataset.data_vars.items():
            if (name in variables and
                z_dim in var.sizes and
                y_dim in var.sizes and
                x_dim in var.sizes):
                updates[name] = var.where(mask_bool.values, mask_value)

        if updates:
            dataset = dataset.assign(**updates)

        return dataset

    except Exception as e:
        raise DataProcessingError("staggered variable terrain masking", str(e))

def center_staggered_variables(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo,
    terrain_mask: xr.DataArray,
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
        terrain_mask: 3D terrain mask array
        center_suffix: Suffix for centered variables

    Returns:
        tuple: (updated_dataset, list_of_new_variable_names)
    """
    from ..core.config import STAGGERED_VARIABLES, STAGGER_CONFIG, WIND_VARIABLES, VORTICITY_VARIABLES

    vertical_dim = VERTICAL_DIM
    x_dim, y_dim = coord_info.x_dim, coord_info.y_dim

    # Check which staggered variables are present
    available_vars = [var for var in STAGGERED_VARIABLES if var in dataset.data_vars]
    if not available_vars:
        return dataset, []

    # Separate winds and vorticities for different masking strategies
    available_winds = [var for var in available_vars if var in WIND_VARIABLES]
    available_vorticities = [var for var in available_vars if var in VORTICITY_VARIABLES]

    # Mask wind variables to 0.0 inside terrain (to avoid affecting averages)
    if available_winds:
        dataset = mask_staggered_variables_for_centering(
            dataset, available_winds, terrain_mask, mask_value=0.0
        )

    # Mask vorticity variables to NaN inside terrain
    if available_vorticities:
        dataset = mask_staggered_variables_for_centering(
            dataset, available_vorticities, terrain_mask, mask_value=np.nan
        )

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

        # Determine fill value based on variable type and dimension
        is_wind = var_name in WIND_VARIABLES

        # Center the variable based on stagger configuration
        if len(stagger_dims) == 1:
            # Single-direction stagger (u, v, w)
            centered = _center_single_direction(
                var_data, stagger_dims[0],
                x_dim, y_dim, vertical_dim,
                slice_info, is_wind
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


def _center_single_direction(
    var: xr.DataArray,
    stagger_dim: str,
    x_dim: str,
    y_dim: str,
    z_dim: str,
    slice_info: SliceInfo,
    is_wind: bool
) -> xr.DataArray:
    """
    Center a variable staggered in a single direction.

    Args:
        var: Variable to center
        stagger_dim: Dimension where variable is staggered ('x', 'y', or 'z')
        x_dim, y_dim, z_dim: Actual dimension names in dataset
        slice_info: Slice information for periodicity
        is_wind: True for wind variables (use 0.0 for boundary), False for vorticity (use NaN)

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
        # Non-periodic: use shift with fill value
        var_shifted = var.shift({actual_dim: 1}).fillna(np.nan)

    centered = 0.5 * (var + var_shifted)
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
        p2 = var.shift({dim1: 1}).fillna(np.nan)

    # Point 3: shifted in dim2 (i, j-1)
    if periodic2:
        p3 = var.roll({dim2: 1}, roll_coords=False)
    else:
        p3 = var.shift({dim2: 1}).fillna(np.nan)

    # Point 4: shifted in both dims (i-1, j-1)
    if periodic1 and periodic2:
        p4 = var.roll({dim1: 1, dim2: 1}, roll_coords=False)
    elif periodic1 and not periodic2:
        p4 = var.roll({dim1: 1}, roll_coords=False).shift({dim2: 1}).fillna(np.nan)
    elif not periodic1 and periodic2:
        p4 = var.shift({dim1: 1}).fillna(np.nan).roll({dim2: 1}, roll_coords=False)
    else:
        p4 = var.shift({dim1: 1, dim2: 1}).fillna(np.nan)

    # Average the 4 points (xarray handles NaN properly in mean)
    # Stack the 4 points along a temporary dimension and take mean
    stacked = xr.concat([p1, p2, p3, p4], dim='_temp_avg')
    centered = stacked.mean(dim='_temp_avg')

    return centered
