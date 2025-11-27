"""
Dynamic Diagnostic Variables

This module implements dynamics-related diagnostic variables including:
- Wind speed (ws)
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
# Exports
# ============================================================================

__all__ = [
    'compute_wind_speed',
]
