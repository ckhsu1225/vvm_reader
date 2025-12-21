"""
Energy Diagnostic Variables

This module implements energy-related diagnostic variables including:
- Dry static energy (sd)
- Moist static energy (hm)
- Saturation moist static energy (hms)
"""

import xarray as xr
from typing import Dict

from .constants import Cp_d, g, Lv
from .registry import register_diagnostic
from .thermodynamics import saturation_mixing_ratio

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# Static Energy
# ============================================================================

@register_diagnostic(
    name='sd',
    profile_dependencies=['PIBAR'],
    diagnostic_dependencies=['t'],
    long_name='dry static energy',
    units='J kg-1',
    description='dry static energy (Cp*T + g*z)',
)
def compute_dry_static_energy(ds: xr.Dataset, profiles: xr.Dataset,
                               diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute dry static energy.

    Formula:
        DSE = Cp × T + g × z

    Args:
        ds: Dataset with vertical coordinate 'lev' (height in m)
        profiles: Dictionary containing reference profiles
        diagnostics: Dictionary containing 'T' (temperature)

    Returns:
        Dry static energy [J/kg]
    """
    T = diagnostics['t']

    # Get height coordinate
    if 'lev' in ds.coords:
        z = ds.coords['lev']
    elif 'lev' in T.coords:
        z = T.coords['lev']
    else:
        logger.error("Vertical coordinate 'lev' not found for DSE calculation")
        return None

    # Compute DSE
    DSE = Cp_d * T + g * z

    DSE.attrs = {
        'long_name': 'dry static energy',
        'units': 'J kg-1',
        'description': 'Cp*T + g*z'
    }

    return DSE


@register_diagnostic(
    name='hm',
    file_dependencies=['qv'],
    profile_dependencies=['PIBAR'],
    diagnostic_dependencies=['t'],
    long_name='moist static energy',
    units='J kg-1',
    description='moist static energy including latent heat from vapor and ice',
)
def compute_moist_static_energy(ds: xr.Dataset, profiles: xr.Dataset,
                                 diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute moist static energy.

    Automatically includes ice phase latent heat effects if qi is present
    in the dataset. The reference state is liquid water at 0°C.

    Energy contributions (relative to liquid water at 0°C):
        - Water vapor (qv): +L_v (energy needed to evaporate from liquid)
        - Liquid water (qc, qr): 0 (reference state, not included in formula)
        - Ice (qi): -L_f (energy released during freezing)

    Formula:
        MSE = Cp × T + g × z + L_v × qv - L_f × qi

    Args:
        ds: Dataset containing 'qv' and optionally 'qi', with vertical coordinate 'lev'
        profiles: Dictionary containing reference profiles
        diagnostics: Dictionary containing 'T'

    Returns:
        Moist static energy [J/kg]
    """
    T = diagnostics['t']
    qv = ds['qv']

    # Get height coordinate
    if 'lev' in ds.coords:
        z = ds.coords['lev']
    elif 'lev' in T.coords:
        z = T.coords['lev']
    else:
        logger.error("Vertical coordinate 'lev' not found for MSE calculation")
        return None

    # Vapor latent heat (always included)
    latent_heat_term = Lv * qv

    # Ice latent heat (automatic detection)
    # Ice has less energy than liquid water (freezing releases L_f)
    q_ice = 0.0
    ice_components = []

    for var in ['qi']:
        if var in ds.data_vars:
            q_ice = q_ice + ds[var]
            ice_components.append(var)

    # Subtract ice latent heat (ice has less energy than liquid water)
    if ice_components:
        from .constants import Lf
        latent_heat_term = latent_heat_term - Lf * q_ice
        ice_status = f"included ({', '.join(ice_components)})"
        logger.debug(f"MSE: Including ice phase effects from {ice_components}")
    else:
        ice_status = "not available"

    # Compute MSE
    MSE = Cp_d * T + g * z + latent_heat_term

    MSE.attrs = {
        'long_name': 'moist static energy',
        'units': 'J kg-1',
        'description': f'Cp*T + g*z + Lv*qv - Lf*q_ice; ice phase {ice_status}',
        'reference_state': 'liquid water at 0°C'
    }

    return MSE


@register_diagnostic(
    name='hms',
    profile_dependencies=['PIBAR', 'PBAR'],
    diagnostic_dependencies=['t'],
    long_name='saturation moist static energy',
    units='J kg-1',
    description='moist static energy at saturation with temperature-dependent latent heat',
)
def compute_saturation_moist_static_energy(ds: xr.Dataset, profiles: xr.Dataset,
                                            diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute saturation moist static energy.

    This represents the moist static energy an air parcel would have if it were
    saturated and all moisture condensed, releasing latent heat.

    Physical interpretation:
        - Initial state: Air parcel at saturation with mixing ratio qs (all vapor)
        - Final state: All vapor condenses to liquid/ice (depending on temperature)
        - Released latent heat: L_mix (temperature-dependent)

    Temperature-dependent latent heat:
        - T > 0°C: vapor → liquid water, releases L_v
        - T < -23°C: vapor → ice, releases L_s = L_v + L_f
        - -23°C < T < 0°C: mixed phase, releases L_mix = L_v + (1-weight)*L_f

    Formula:
        MSE_s = Cp × T + g × z + L_mix × qs

    Args:
        ds: Dataset with vertical coordinate 'lev'
        profiles: Dictionary containing 'PBAR'
        diagnostics: Dictionary containing 'T'

    Returns:
        Saturation moist static energy [J/kg]
    """
    T = diagnostics['t']

    # Get height coordinate
    if 'lev' in ds.coords:
        z = ds.coords['lev']
    elif 'lev' in T.coords:
        z = T.coords['lev']
    else:
        logger.error("Vertical coordinate 'lev' not found for MSE_s calculation")
        return None

    # profiles['PBAR'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    P = profiles['PBAR']

    # Compute saturation mixing ratio (temperature-dependent, mixed-phase)
    qs = saturation_mixing_ratio(T, P)

    # Compute temperature-dependent effective latent heat
    # weight = 0 means pure ice (L_mix = Ls), weight = 1 means pure liquid (L_mix = Lv)
    from .constants import T_ice_threshold, T_freeze, Lf

    weight = (T - T_ice_threshold) / (T_freeze - T_ice_threshold)
    weight = weight.clip(0.0, 1.0)

    # L_mix: blended latent heat
    # At T < -23°C: L_mix = Ls (sublimation)
    # At T > 0°C: L_mix = Lv (condensation)
    # In between: linear blend
    L_mix = Lv + (1.0 - weight) * Lf  # Equivalent to: weight*Lv + (1-weight)*Ls

    # Compute MSE_s with temperature-dependent latent heat
    latent_heat_term = L_mix * qs

    # Compute MSE_s
    MSE_s = Cp_d * T + g * z + latent_heat_term

    # Determine phase status for metadata
    ice_fraction = 1.0 - weight
    has_ice = (ice_fraction > 0).any()

    if has_ice:
        logger.debug("MSE_s: Using temperature-dependent L_mix (includes ice phase)")

    MSE_s.attrs = {
        'long_name': 'saturation moist static energy',
        'units': 'J kg-1',
        'description': f'Cp*T + g*z + L_mix*qs at saturation; phase {"temperature-dependent (includes ice)" if has_ice else "pure liquid water"}',
        'reference_state': 'liquid water at 0°C'
    }

    return MSE_s


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'compute_dry_static_energy',
    'compute_moist_static_energy',
    'compute_saturation_moist_static_energy',
]
