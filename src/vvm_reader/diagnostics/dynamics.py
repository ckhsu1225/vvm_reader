"""
Dynamic Diagnostic Variables

This module implements dynamics-related diagnostic variables including:
- Wind speed (ws)
- Potential vorticity (pv)
"""

import numpy as np
import xarray as xr
from typing import Dict

from .registry import register_diagnostic

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# Wind Diagnostics
# ============================================================================

@register_diagnostic(
    name='ws',
    file_dependencies=['u', 'v'],
    long_name='wind speed',
    units='m s-1',
    description='horizontal wind speed calculated from u and v components',
    standard_name='wind_speed'
)
def compute_wind_speed(ds: xr.Dataset, profiles: xr.Dataset,  # noqa: ARG001
                       diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:  # noqa: ARG001
    """
    Compute horizontal wind speed from u and v components.

    Formula:
        ws = sqrt(u^2 + v^2)

    Args:
        ds: Dataset containing 'u' (eastward wind) and 'v' (northward wind)
        profiles: Dataset containing reference profiles (not used)
        diagnostics: Dictionary of previously computed diagnostic variables (not used)

    Returns:
        Wind speed [m/s]

    Note:
        This computes only horizontal wind speed. For 3D wind speed including
        vertical component w, use: sqrt(u^2 + v^2 + w^2).
    """
    u = ds['u']
    v = ds['v']

    # Compute horizontal wind speed
    ws = np.sqrt(u**2 + v**2)

    ws.attrs = {
        'long_name': 'wind speed',
        'units': 'm s-1',
        'standard_name': 'wind_speed',
        'description': 'horizontal wind speed (magnitude of horizontal wind vector)'
    }

    return ws


# ============================================================================
# Potential Vorticity
# ============================================================================

@register_diagnostic(
    name='pv',
    file_dependencies=['zeta', 'eta', 'xi', 'th'],
    profile_dependencies=['RHO'],
    long_name='Ertel potential vorticity',
    units='K m2 kg-1 s-1',
    description='3D Ertel potential vorticity',
    standard_name='ertel_potential_vorticity'
)
def compute_potential_vorticity(ds: xr.Dataset, profiles: xr.Dataset,
                                 diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:  # noqa: ARG001
    """
    Compute 3D Ertel Potential Vorticity.

    Formula:
        PV = (1/ρ) × [ξ×(∂θ/∂x) + η×(∂θ/∂y) + (ζ+f)×(∂θ/∂z)]

    where:
        - ξ, η, ζ are the vorticity components (x, y, z) from VVM output
        - f is the Coriolis parameter (from ds.attrs['coriolis_parameter'])
        - θ is the potential temperature
        - ρ is the background air density from profiles

    Args:
        ds: Dataset containing 'zeta', 'eta', 'xi', 'th'
            Must have 'coriolis_parameter' in attrs if f-plane is used
        profiles: Dataset containing 'RHO'
        diagnostics: Dictionary of previously computed diagnostic variables (not used)

    Returns:
        Ertel potential vorticity [K m² kg⁻¹ s⁻¹]

    Notes:
        - If Coriolis parameter is not available, f=0 is used (non-rotating simulation)
        - Typical PV units are 1 PVU = 1e-6 K m² kg⁻¹ s⁻¹
        - VVM outputs vorticity components directly, no need to compute from wind
        - VVM's eta output has opposite sign from standard definition; corrected here
    """
    # Get vorticity components
    zeta = ds['zeta']  # vertical component (∂v/∂x - ∂u/∂y)
    eta = -ds['eta']   # y-component: VVM output has opposite sign, corrected here
    xi = ds['xi']      # x-component (∂w/∂y - ∂v/∂z)

    # Get potential temperature
    th = ds['th']

    # Get Coriolis parameter from dataset attributes
    # If None, assume non-rotating simulation (f=0)
    f = ds.attrs.get('coriolis_parameter', None)
    if f is None:
        logger.info(
            "Coriolis parameter not available. Using f=0 (non-rotating simulation). "
            "To include f, enable add_coriolis_parameter=True in ProcessingOptions."
        )
        f = 0.0

    # Get background density
    rho = profiles['RHO']

    # Ensure lev dimension has sufficient chunk size for differentiation
    # xarray.differentiate uses edge_order=2 by default, requiring chunk size >= 4
    MIN_CHUNK_SIZE = 4
    if hasattr(th.data, 'chunks'):
        lev_dim_idx = th.dims.index('lev') if 'lev' in th.dims else None
        if lev_dim_idx is not None:
            lev_chunks = th.data.chunks[lev_dim_idx]
            if any(c < MIN_CHUNK_SIZE for c in lev_chunks):
                logger.debug(
                    f"Rechunking lev dimension for differentiation "
                    f"(current chunks: {lev_chunks}, minimum: {MIN_CHUNK_SIZE})"
                )
                th = th.chunk({'lev': -1})

    th['xc'] = th['xc'].compute()
    th['yc'] = th['yc'].compute()

    # Compute potential temperature gradients using xc, yc (meters) for horizontal
    # and lev (meters) for vertical differentiation
    dth_dx = th.differentiate('xc')
    dth_dy = th.differentiate('yc')
    dth_dz = th.differentiate('lev')

    # Compute absolute vorticity (vertical component)
    zeta_abs = zeta + f

    # Compute Ertel PV
    # PV = (1/ρ) × [ξ×(∂θ/∂x) + η×(∂θ/∂y) + (ζ+f)×(∂θ/∂z)]
    PV = (1.0 / rho) * (xi * dth_dx + eta * dth_dy + zeta_abs * dth_dz)

    PV.attrs = {
        'long_name': 'Ertel potential vorticity',
        'units': 'K m2 kg-1 s-1',
        'standard_name': 'ertel_potential_vorticity',
        'description': '3D Ertel potential vorticity using VVM vorticity outputs'
    }

    return PV


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'compute_wind_speed',
    'compute_potential_vorticity',
]
