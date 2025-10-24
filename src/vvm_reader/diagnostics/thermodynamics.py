"""
Thermodynamic Diagnostic Variables

This module implements thermodynamic diagnostic variables including:
- Temperature (T)
- Virtual temperature (T_v)
- Virtual potential temperature (theta_v)
- Equivalent potential temperature (theta_e)
- Saturation equivalent potential temperature (theta_es)

All calculations use PIBAR as the background Exner function, which is
appropriate for most applications with errors <2%.
"""

import numpy as np
import xarray as xr
from typing import Dict

from .constants import (
    Cp_d, P0, Lv, Lf, epsilon, kappa,
    es0_liquid, a_liquid, b_liquid,
    es0_ice, a_ice, b_ice, T_freeze, T_ice_threshold,
    kelvin_to_celsius
)
from .registry import register_diagnostic

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# Saturation Vapor Pressure
# ============================================================================

def saturation_vapor_pressure(T: xr.DataArray, over_ice: bool = None) -> xr.DataArray:
    """
    Calculate saturation vapor pressure using August-Roche-Magnus formula.

    Args:
        T: Temperature [K]
        over_ice: If True, force ice formula; if False, force liquid water formula.
                  If None (default), automatically determine based on temperature:
                  - T < 250.15 K (-23°C): use ice formula
                  - T > 273.15 K (0°C): use liquid water formula
                  - 250.15 K ≤ T ≤ 273.15 K: linear blend (mixed-phase)

    Returns:
        Saturation vapor pressure [Pa]

    References:
        - Alduchov & Eskridge (1996): Improved Magnus Form Approximation of
          Saturation Vapor Pressure, J. Appl. Meteor., 35, 601-609
        - Pruppacher & Klett (1997): Microphysics of Clouds and Precipitation
        - ECMWF IFS Documentation (Cy47r3, 2021): Mixed-phase parametrization

    Note:
        The automatic mode uses a temperature-dependent blend following common
        practice in NWP models (ECMWF, GFS). The -23°C threshold is based on
        observations showing that supercooled liquid water droplets become rare
        at this temperature in natural clouds. At colder temperatures, ice
        crystals dominate; at warmer temperatures near 0°C, liquid water dominates.
    """
    T_celsius = kelvin_to_celsius(T)

    if over_ice is True:
        # Forced ice formula
        es = es0_ice * np.exp(a_ice * T_celsius / (T_celsius + b_ice))
        description = 'Saturation vapor pressure over ice'

    elif over_ice is False:
        # Forced liquid water formula
        es = es0_liquid * np.exp(a_liquid * T_celsius / (T_celsius + b_liquid))
        description = 'Saturation vapor pressure over liquid water'

    else:
        # Automatic mode: temperature-dependent
        # Compute both formulas
        es_ice = es0_ice * np.exp(a_ice * T_celsius / (T_celsius + b_ice))
        es_liquid = es0_liquid * np.exp(a_liquid * T_celsius / (T_celsius + b_liquid))

        # Use xarray's where to apply temperature-dependent formula
        # T < T_ice_threshold (-23°C): pure ice
        # T > T_freeze (0°C): pure liquid water
        # In between: linear blend (mixed phase)

        # Calculate blending weight (0 = ice, 1 = liquid)
        # weight = (T - T_ice_threshold) / (T_freeze - T_ice_threshold)
        weight = (T - T_ice_threshold) / (T_freeze - T_ice_threshold)
        weight = weight.clip(0.0, 1.0)  # Ensure 0 ≤ weight ≤ 1

        # Blend between ice and liquid formulas
        es = (1.0 - weight) * es_ice + weight * es_liquid
        description = 'Saturation vapor pressure (temperature-dependent, mixed-phase)'

    # Preserve metadata
    es_out = xr.DataArray(
        es,
        coords=T.coords,
        dims=T.dims,
        attrs={
            'long_name': 'saturation vapor pressure',
            'units': 'Pa',
            'description': description
        }
    )

    return es_out


def saturation_mixing_ratio(T: xr.DataArray, P: xr.DataArray, over_ice: bool = None) -> xr.DataArray:
    """
    Calculate saturation mixing ratio.

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        over_ice: If True, force ice formula; if False, force liquid water formula.
                  If None (default), automatically determine based on temperature.

    Returns:
        Saturation mixing ratio [kg/kg]

    Formula:
        qs = epsilon * es / (P - es)
        ≈ epsilon * es / P  (when es << P)
    """
    es = saturation_vapor_pressure(T, over_ice=over_ice)

    # Full formula (more accurate)
    qs = epsilon * es / (P - es)

    # Handle potential division issues
    qs = qs.where(P > es, other=np.nan)

    # Set description based on mode
    if over_ice is True:
        desc = 'Saturation mixing ratio over ice'
    elif over_ice is False:
        desc = 'Saturation mixing ratio over liquid water'
    else:
        desc = 'Saturation mixing ratio (temperature-dependent, mixed-phase)'

    qs.attrs = {
        'long_name': 'saturation mixing ratio',
        'units': 'kg kg-1',
        'description': desc
    }

    return qs


# ============================================================================
# Basic Temperature Diagnostics
# ============================================================================

@register_diagnostic(
    name='T',
    file_dependencies=['th'],
    profile_dependencies=['PIBAR'],
    long_name='Temperature',
    units='K',
    description='Absolute temperature computed from potential temperature using background Exner function',
    standard_name='air_temperature'
)
def compute_temperature(ds: xr.Dataset, profiles: xr.Dataset,
                        diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute temperature from potential temperature.

    Formula:
        T = θ × π
        where π ≈ PIBAR(z) (background Exner function)

    Args:
        ds: Dataset containing 'th' (potential temperature)
        profiles: Dataset containing 'PIBAR' (Exner function profile)
        diagnostics: Dictionary of previously computed diagnostic variables

    Returns:
        Temperature [K]

    Note:
        This uses PIBAR as background Exner function, which neglects
        perturbation π'. Typical errors are <2% for most applications.
        xarray automatically aligns the lev coordinate between th and PIBAR.
    """
    th = ds['th']

    # profiles['PIBAR'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates during multiplication
    T = th * profiles['PIBAR']

    T.attrs = {
        'long_name': 'Temperature',
        'units': 'K',
        'standard_name': 'air_temperature',
        'description': 'Computed from theta using background Exner function PIBAR'
    }

    return T


@register_diagnostic(
    name='T_v',
    file_dependencies=['th', 'qv'],
    profile_dependencies=['PIBAR'],
    diagnostic_dependencies=['T'],
    long_name='Virtual temperature',
    units='K',
    description='Virtual temperature accounting for moisture and hydrometeor effects on air density',
    standard_name='virtual_temperature'
)
def compute_virtual_temperature(ds: xr.Dataset, profiles: xr.Dataset,
                                 diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute virtual temperature.

    Automatically includes all available hydrometeors (qc, qi, qr, qrim) in the
    calculation. Hydrometeors increase air density without contributing to pressure,
    thus affecting buoyancy.

    Formula:
        T_v = T × (1 + qv/ε) / (1 + qv + qc + qi + qr + qrim)

    Or equivalently:
        T_v = T × (1 + (1/ε - 1) × qv - q_condensate)

    Args:
        ds: Dataset containing 'qv' and optionally 'qc', 'qi', 'qr', 'qrim'
        profiles: Dataset containing reference profiles
        diagnostics: Dictionary containing 'T' (temperature)

    Returns:
        Virtual temperature [K]
    """
    T = diagnostics['T']
    qv = ds['qv']

    # Total condensate (all hydrometeors except vapor)
    q_condensate = 0.0
    condensate_components = []

    for var in ['qc', 'qi', 'qr', 'qrim']:
        if var in ds.data_vars:
            q_condensate = q_condensate + ds[var]
            condensate_components.append(var)

    # Virtual temperature formula including all water species
    T_v = T * (1.0 + qv / epsilon) / (1.0 + qv + q_condensate)

    # Document what was included
    if condensate_components:
        hydrometeor_status = f"included ({', '.join(condensate_components)})"
        logger.debug(f"T_v: Including hydrometeors: {condensate_components}")
    else:
        hydrometeor_status = "not available (vapor only)"

    T_v.attrs = {
        'long_name': 'Virtual temperature',
        'units': 'K',
        'standard_name': 'virtual_temperature',
        'description': f'Temperature of dry air with same density; hydrometeors {hydrometeor_status}'
    }

    return T_v


@register_diagnostic(
    name='theta_v',
    file_dependencies=['th', 'qv'],
    long_name='Virtual potential temperature',
    units='K',
    description='Potential temperature accounting for moisture and hydrometeor effects',
)
def compute_virtual_potential_temperature(ds: xr.Dataset, profiles: xr.Dataset,
                                          diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute virtual potential temperature.

    Automatically includes all available hydrometeors (qc, qi, qr, qrim) in the
    calculation. This is the potential temperature that dry air would have to
    have the same density as the moist air with all hydrometeors.

    Formula:
        θ_v = θ × (1 + qv/ε) / (1 + qv + qc + qi + qr + qrim)

    Args:
        ds: Dataset containing 'th', 'qv' and optionally 'qc', 'qi', 'qr', 'qrim'
        profiles: Dataset containing reference profiles
        diagnostics: Dictionary of previously computed diagnostic variables

    Returns:
        Virtual potential temperature [K]
    """
    th = ds['th']
    qv = ds['qv']

    # Total condensate (all hydrometeors except vapor)
    q_condensate = 0.0
    condensate_components = []

    for var in ['qc', 'qi', 'qr', 'qrim']:
        if var in ds.data_vars:
            q_condensate = q_condensate + ds[var]
            condensate_components.append(var)

    # Virtual potential temperature formula including all water species
    theta_v = th * (1.0 + qv / epsilon) / (1.0 + qv + q_condensate)

    # Document what was included
    if condensate_components:
        hydrometeor_status = f"included ({', '.join(condensate_components)})"
        logger.debug(f"theta_v: Including hydrometeors: {condensate_components}")
    else:
        hydrometeor_status = "not available (vapor only)"

    theta_v.attrs = {
        'long_name': 'Virtual potential temperature',
        'units': 'K',
        'description': f'Potential temperature of dry air with same density; hydrometeors {hydrometeor_status}'
    }

    return theta_v


# ============================================================================
# Equivalent Potential Temperature
# ============================================================================

@register_diagnostic(
    name='theta_e',
    file_dependencies=['th', 'qv'],
    profile_dependencies=['PIBAR', 'PBAR'],
    diagnostic_dependencies=['T'],
    long_name='Equivalent potential temperature',
    units='K',
    description='Potential temperature that air would have if all moisture condensed',
)
def compute_equivalent_potential_temperature(ds: xr.Dataset, profiles: xr.Dataset,
                                             diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute equivalent potential temperature using Bolton's formula.

    Automatically includes ice phase latent heat effects if qi and/or qrim
    are present in the dataset. The reference state is liquid water at 0°C.

    Energy contributions (relative to liquid water at 0°C):
        - Water vapor (qv): +L_v (energy needed to evaporate from liquid)
        - Liquid water (qc, qr): 0 (reference state, not included in formula)
        - Ice (qi, qrim): -L_f (energy released during freezing)

    Formula:
        θ_e = θ × exp((L_v × qv - L_f × q_ice) / (Cp × T))

    Args:
        ds: Dataset containing 'th', 'qv' and optionally 'qi', 'qrim'
        profiles: Dataset containing reference profiles
        diagnostics: Dictionary containing 'T'

    Returns:
        Equivalent potential temperature [K]

    References:
        Bolton, D. (1980): The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046-1053.
    """
    th = ds['th']
    qv = ds['qv']
    T = diagnostics['T']

    # Vapor latent heat (always included)
    latent_term = Lv * qv / (Cp_d * T)

    # Ice latent heat (automatic detection)
    # Ice has less energy than liquid water (freezing releases L_f)
    q_ice = 0.0
    ice_components = []

    for var in ['qi', 'qrim']:
        if var in ds.data_vars:
            q_ice = q_ice + ds[var]
            ice_components.append(var)

    # Subtract ice latent heat (ice has less energy than liquid water)
    if ice_components:
        latent_term = latent_term - Lf * q_ice / (Cp_d * T)
        ice_status = f"included ({', '.join(ice_components)})"
        logger.debug(f"theta_e: Including ice phase effects from {ice_components}")
    else:
        ice_status = "not available"

    theta_e = th * np.exp(latent_term)

    theta_e.attrs = {
        'long_name': 'Equivalent potential temperature',
        'units': 'K',
        'description': f'Bolton (1980) formula; ice phase {ice_status}',
        'reference_state': 'liquid water at 0°C'
    }

    return theta_e


@register_diagnostic(
    name='theta_es',
    file_dependencies=['th'],
    profile_dependencies=['PIBAR', 'PBAR'],
    diagnostic_dependencies=['T'],
    long_name='Saturation equivalent potential temperature',
    units='K',
    description='Equivalent potential temperature at saturation',
)
def compute_saturation_equivalent_potential_temperature(
    ds: xr.Dataset,
    profiles: xr.Dataset,
    diagnostics: Dict[str, xr.DataArray]
) -> xr.DataArray:
    """
    Compute saturation equivalent potential temperature using Bolton's formula.

    This represents the potential temperature an air parcel would have if it were
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
        θ_es = θ × exp(L_mix × qs / (Cp × T))

    Args:
        ds: Dataset containing 'th'
        profiles: Dataset containing 'PIBAR', 'PBAR'
        diagnostics: Dictionary containing 'T'

    Returns:
        Saturation equivalent potential temperature [K]

    References:
        Bolton, D. (1980): The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046-1053.
    """
    th = ds['th']
    T = diagnostics['T']

    # profiles['PBAR'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    P = profiles['PBAR']

    # Get saturation mixing ratio (temperature-dependent, mixed-phase)
    qs = saturation_mixing_ratio(T, P)

    # Compute temperature-dependent effective latent heat
    # weight = 0 means pure ice (L_mix = Ls), weight = 1 means pure liquid (L_mix = Lv)
    weight = (T - T_ice_threshold) / (T_freeze - T_ice_threshold)
    weight = weight.clip(0.0, 1.0)

    # L_mix: blended latent heat
    # At T < -23°C: L_mix = Ls (sublimation)
    # At T > 0°C: L_mix = Lv (condensation)
    # In between: linear blend
    L_mix = Lv + (1.0 - weight) * Lf  # Equivalent to: weight*Lv + (1-weight)*Ls

    # Compute theta_es with temperature-dependent latent heat
    latent_term = L_mix * qs / (Cp_d * T)
    theta_es = th * np.exp(latent_term)

    # Determine phase status for metadata
    ice_fraction = 1.0 - weight
    has_ice = (ice_fraction > 0).any()

    if has_ice:
        logger.debug("theta_es: Using temperature-dependent L_mix (includes ice phase)")

    theta_es.attrs = {
        'long_name': 'Saturation equivalent potential temperature',
        'units': 'K',
        'description': f'Bolton (1980) formula at saturation with L_mix; phase {"temperature-dependent (includes ice)" if has_ice else "pure liquid water"}',
        'reference_state': 'liquid water at 0°C'
    }

    return theta_es


# ============================================================================
# Utility Functions
# ============================================================================

def compute_pressure_from_exner(pi: xr.DataArray) -> xr.DataArray:
    """
    Compute pressure from Exner function.

    Formula:
        π = (P / P0)^(R/Cp)
        => P = P0 × π^(Cp/R)

    Args:
        pi: Exner function [dimensionless]

    Returns:
        Pressure [Pa]
    """
    P = P0 * pi ** (1.0 / kappa)

    P.attrs = {
        'long_name': 'Pressure',
        'units': 'Pa',
        'description': 'Computed from Exner function'
    }

    return P


def compute_exner_from_pressure(P: xr.DataArray) -> xr.DataArray:
    """
    Compute Exner function from pressure.

    Formula:
        π = (P / P0)^(R/Cp)

    Args:
        P: Pressure [Pa]

    Returns:
        Exner function [dimensionless]
    """
    pi = (P / P0) ** kappa

    pi.attrs = {
        'long_name': 'Exner function',
        'units': 'dimensionless',
        'description': 'Computed from pressure'
    }

    return pi


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Registered diagnostics
    'compute_temperature',
    'compute_virtual_temperature',
    'compute_virtual_potential_temperature',
    'compute_equivalent_potential_temperature',
    'compute_saturation_equivalent_potential_temperature',

    # Utility functions
    'saturation_vapor_pressure',
    'saturation_mixing_ratio',
    'compute_pressure_from_exner',
    'compute_exner_from_pressure',
]
