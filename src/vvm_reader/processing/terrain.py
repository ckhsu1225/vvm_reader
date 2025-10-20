"""
VVM Reader Terrain Processing

This module handles terrain masking and related topography operations.
"""

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

def mask_wind_variables_for_centering(
    dataset: xr.Dataset,
    wind_variables: list,
    terrain_mask: xr.DataArray
) -> xr.Dataset:
    """
    Mask wind variables inside terrain before centering operations.
    
    Args:
        dataset: Input dataset
        wind_variables: List of wind variable names to mask
        terrain_mask: 3D terrain mask array
    
    Returns:
        xr.Dataset: Dataset with wind variables masked inside terrain
    """
    try:
        z_dim, y_dim, x_dim = terrain_mask.sizes
        mask_bool = terrain_mask.astype(bool)

        updates = {}
        for name, var in dataset.data_vars.items():
            if (name in wind_variables and
                z_dim in var.sizes and
                y_dim in var.sizes and
                x_dim in var.sizes):
                updates[name] = var.where(mask_bool.values, 0.0)

        if updates:
            dataset = dataset.assign(**updates)

        return dataset

    except Exception as e:
        raise DataProcessingError("wind terrain masking", str(e))

def center_staggered_winds(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo,
    terrain_mask: xr.DataArray,
    center_suffix: str = "_c"
) -> tuple:
    """
    Center staggered wind variables to cell centers with terrain awareness.
    
    Centering rules:
    - u: avg of u(i) and u(i-1), periodic only for full domain
    - v: avg of v(j) and v(j-1), periodic only for full domain  
    - w: avg of w(k) and w(k-1), non-periodic (bottom ghost = 0)
    
    Args:
        dataset: Input dataset
        coord_info: Coordinate information
        slice_info: Slice information for periodicity
        terrain_mask: 3D terrain mask array
        center_suffix: Suffix for centered variables
        
    Returns:
        tuple: (updated_dataset, list_of_new_variable_names)
    """
    from ..core.config import WIND_VARIABLES
    
    vertical_dim = VERTICAL_DIM
    x_dim, y_dim = coord_info.x_dim, coord_info.y_dim
    
    # Check which wind variables are present
    available_winds = [var for var in WIND_VARIABLES if var in dataset.data_vars]
    if not available_winds:
        return dataset, []
    
    # Mask wind variables to 0.0 inside terrain before centering
    dataset = mask_wind_variables_for_centering(
        dataset, available_winds, terrain_mask
    )
    
    new_variables = {}
    new_var_names = []
    
    # Center u-wind (x-direction)
    if "u" in dataset.data_vars:
        u_var = dataset["u"]
        if y_dim in u_var.sizes and x_dim in u_var.sizes:
            if slice_info.periodic_x:
                # Periodic centering: roll by 1
                u_centered = 0.5 * (u_var + u_var.roll({x_dim: 1}, roll_coords=False))
            else:
                # Non-periodic centering: shift by 1
                u_centered = 0.5 * (u_var + u_var.shift({x_dim: 1}))
            
            # Preserve attributes and encoding
            u_centered.attrs.update(u_var.attrs)
            u_centered.encoding = getattr(u_var, 'encoding', {})
            
            var_name = f"u{center_suffix}" if center_suffix else "u"
            new_variables[var_name] = u_centered
            new_var_names.append(var_name)
    
    # Center v-wind (y-direction)
    if "v" in dataset.data_vars:
        v_var = dataset["v"]
        if y_dim in v_var.sizes and x_dim in v_var.sizes:
            if slice_info.periodic_y:
                # Periodic centering: roll by 1
                v_centered = 0.5 * (v_var + v_var.roll({y_dim: 1}, roll_coords=False))
            else:
                # Non-periodic centering: shift by 1
                v_centered = 0.5 * (v_var + v_var.shift({y_dim: 1}))
            
            # Preserve attributes and encoding
            v_centered.attrs.update(v_var.attrs)
            v_centered.encoding = getattr(v_var, 'encoding', {})
            
            var_name = f"v{center_suffix}" if center_suffix else "v"
            new_variables[var_name] = v_centered
            new_var_names.append(var_name)
    
    # Center w-wind (vertical direction)
    if "w" in dataset.data_vars:
        w_var = dataset["w"]
        if vertical_dim in w_var.sizes and y_dim in w_var.sizes and x_dim in w_var.sizes:
            # Non-periodic vertical centering with bottom ghost level = 0
            w_below = w_var.shift({vertical_dim: 1}).fillna(0.0)
            w_centered = 0.5 * (w_var + w_below)
            
            # Preserve attributes and encoding
            w_centered.attrs.update(w_var.attrs)
            w_centered.encoding = getattr(w_var, 'encoding', {})
            
            var_name = f"w{center_suffix}" if center_suffix else "w"
            new_variables[var_name] = w_centered
            new_var_names.append(var_name)
    
    if new_variables:
        dataset = dataset.assign(**new_variables)
    
    return dataset, new_var_names
