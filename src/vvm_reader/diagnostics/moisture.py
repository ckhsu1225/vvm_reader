"""
Moisture Diagnostic Variables

This module implements moisture-related diagnostic variables including:
- Relative humidity (rh)
- Saturation mixing ratio (qvs)
- Column water vapor (cwv)
- Liquid water path (lwp)
- Ice water path (iwp)
"""

import xarray as xr
from typing import Dict

from .registry import register_diagnostic
from .thermodynamics import saturation_mixing_ratio

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# Relative Humidity
# ============================================================================

@register_diagnostic(
    name='rh',
    file_dependencies=['qv'],
    profile_dependencies=['PIBAR', 'PBAR'],
    diagnostic_dependencies=['t'],
    long_name='relative humidity',
    units='%',
    description='relative humidity with respect to liquid water',
    standard_name='relative_humidity'
)
def compute_relative_humidity(ds: xr.Dataset, profiles: xr.Dataset,
                               diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute relative humidity.

    Formula:
        RH = (qv / qs) × 100%

    where qs is the saturation mixing ratio at temperature T and pressure P.

    Args:
        ds: Dataset containing 'qv' (water vapor mixing ratio)
        profiles: Dataset containing 'PBAR' (pressure profile)
        diagnostics: Dictionary containing 'T' (temperature)

    Returns:
        Relative humidity [%]
    """
    qv = ds['qv']
    T = diagnostics['t']

    # profiles['PBAR'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    P = profiles['PBAR']

    # Compute saturation mixing ratio (temperature-dependent)
    qs = saturation_mixing_ratio(T, P)

    # Compute RH (cap at 100% to handle numerical issues)
    RH = (qv / qs) * 100.0
    RH = RH.clip(min=0., max=100.0)

    RH.attrs = {
        'long_name': 'relative humidity',
        'units': '%',
        'standard_name': 'relative_humidity',
        'description': 'relative humidity with respect to liquid water'
    }

    return RH


@register_diagnostic(
    name='qvs',
    profile_dependencies=['PIBAR', 'PBAR'],
    diagnostic_dependencies=['t'],
    long_name='saturation mixing ratio',
    units='kg kg-1',
    description='saturation water vapor mixing ratio',
)
def compute_saturation_mixing_ratio(ds: xr.Dataset, profiles: xr.Dataset,
                                    diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute saturation mixing ratio as a diagnostic variable.

    Args:
        ds: Dataset
        profiles: Dataset containing 'PBAR'
        diagnostics: Dictionary containing 'T'

    Returns:
        Saturation mixing ratio [kg/kg]
    """
    T = diagnostics['t']

    # profiles['PBAR'] is xr.DataArray with lev coordinate
    P = profiles['PBAR']

    # Compute saturation mixing ratio (temperature-dependent)
    qs = saturation_mixing_ratio(T, P)

    qs.attrs = {
        'long_name': 'saturation mixing ratio',
        'units': 'kg kg-1',
        'description': 'saturation water vapor mixing ratio over liquid water'
    }

    return qs


# ============================================================================
# Column Integrated Variables
# ============================================================================

@register_diagnostic(
    name='cwv',
    file_dependencies=['qv'],
    profile_dependencies=['RHO'],
    long_name='column water vapor',
    units='kg m-2',
    description='vertically integrated water vapor (precipitable water)',
    standard_name='atmosphere_mass_content_of_water_vapor'
)
def compute_column_water_vapor(ds: xr.Dataset, profiles: xr.Dataset,
                               diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute column water vapor (precipitable water).

    Formula:
        CWV = ∫ ρ × qv dz

    where ρ is air density and qv is water vapor mixing ratio.

    Args:
        ds: Dataset containing 'qv' and vertical coordinate 'lev' (height in m)
        profiles: Dataset containing 'RHO' (density profile)
        diagnostics: Dictionary of previously computed diagnostic variables

    Returns:
        Column water vapor [kg/m²]
    """
    qv = ds['qv']

    # Check if vertical dimension exists
    if 'lev' not in qv.dims:
        logger.warning("CWV calculation requires vertical dimension 'lev'")
        return None

    # Get vertical coordinate and verify units (height in meters)
    if 'lev' in ds.coords:
        z = ds.coords['lev']
        # Verify units are in meters
        if 'units' in z.attrs:
            if z.attrs['units'] not in ['m', 'meter', 'meters']:
                logger.warning(f"CWV: Vertical coordinate units are '{z.attrs['units']}', expected 'm'. Results may be incorrect.")
        else:
            logger.warning("CWV: Vertical coordinate 'lev' has no units attribute. Assuming meters.")
    else:
        logger.error("Vertical coordinate 'lev' not found in dataset")
        return None

    # profiles['RHO'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    integrand = profiles['RHO'] * qv

    # Handle terrain masking: NaN values below terrain should not contribute to integral
    # Replace NaN with 0 before integration (terrain has no water vapor)
    integrand = integrand.fillna(0.0)

    # Use trapezoidal integration
    CWV = integrand.integrate('lev')

    CWV.attrs = {
        'long_name': 'column water vapor',
        'units': 'kg m-2',
        'standard_name': 'atmosphere_mass_content_of_water_vapor',
        'description': 'vertically integrated water vapor (precipitable water)'
    }

    return CWV


@register_diagnostic(
    name='lwp',
    file_dependencies=['qc'],
    profile_dependencies=['RHO'],
    long_name='liquid water path',
    units='kg m-2',
    description='vertically integrated cloud liquid water',
)
def compute_liquid_water_path(ds: xr.Dataset, profiles: xr.Dataset,
                              diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute liquid water path (cloud liquid water only).

    Formula:
        LWP = ∫ ρ × qc dz

    where qc is cloud water mixing ratio.

    Args:
        ds: Dataset containing 'qc'
        profiles: Dictionary containing 'RHO'

    Returns:
        Liquid water path [kg/m²]
    """
    # Check if required variables exist
    if 'qc' not in ds:
        logger.warning("LWP calculation requires 'qc'")
        return None

    qc = ds['qc']

    # Check vertical dimension
    if 'lev' not in qc.dims:
        logger.warning("LWP calculation requires vertical dimension 'lev'")
        return None

    # Get vertical coordinate and verify units (height in meters)
    if 'lev' in ds.coords:
        z = ds.coords['lev']
        # Verify units are in meters
        if 'units' in z.attrs:
            if z.attrs['units'] not in ['m', 'meter', 'meters']:
                logger.warning(f"LWP: Vertical coordinate units are '{z.attrs['units']}', expected 'm'. Results may be incorrect.")
        else:
            logger.warning("LWP: Vertical coordinate 'lev' has no units attribute. Assuming meters.")
    else:
        logger.error("Vertical coordinate 'lev' not found")
        return None

    # profiles['RHO'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    integrand = profiles['RHO'] * qc

    # Handle terrain masking: NaN values below terrain should not contribute to integral
    # Replace NaN with 0 before integration (terrain has no liquid water)
    integrand = integrand.fillna(0.0)

    # Use trapezoidal integration
    LWP = integrand.integrate('lev')

    LWP.attrs = {
        'long_name': 'liquid water path',
        'units': 'kg m-2',
        'description': 'vertically integrated cloud liquid water'
    }

    return LWP


@register_diagnostic(
    name='iwp',
    file_dependencies=['qi'],
    profile_dependencies=['RHO'],
    long_name='ice water path',
    units='kg m-2',
    description='vertically integrated ice water',
)
def compute_ice_water_path(ds: xr.Dataset, profiles: xr.Dataset,
                           diagnostics: Dict[str, xr.DataArray]) -> xr.DataArray:
    """
    Compute ice water path.

    Formula:
        IWP = ∫ ρ × qi dz

    where qi is the total ice mixing ratio (includes all ice species).

    Args:
        ds: Dataset containing 'qi'
        profiles: Dictionary containing 'RHO'

    Returns:
        Ice water path [kg/m²]
    """
    # Check if required variables exist
    if 'qi' not in ds:
        logger.warning("IWP calculation requires 'qi'")
        return None

    qi = ds['qi']

    # Check vertical dimension
    if 'lev' not in qi.dims:
        logger.warning("IWP calculation requires vertical dimension 'lev'")
        return None

    # Get vertical coordinate and verify units (height in meters)
    if 'lev' in ds.coords:
        z = ds.coords['lev']
        # Verify units are in meters
        if 'units' in z.attrs:
            if z.attrs['units'] not in ['m', 'meter', 'meters']:
                logger.warning(f"IWP: Vertical coordinate units are '{z.attrs['units']}', expected 'm'. Results may be incorrect.")
        else:
            logger.warning("IWP: Vertical coordinate 'lev' has no units attribute. Assuming meters.")
    else:
        logger.error("Vertical coordinate 'lev' not found")
        return None

    # profiles['RHO'] is xr.DataArray with lev coordinate
    # xarray will automatically align coordinates
    integrand = profiles['RHO'] * qi

    # Handle terrain masking: NaN values below terrain should not contribute to integral
    # Replace NaN with 0 before integration (terrain has no ice water)
    integrand = integrand.fillna(0.0)

    # Use trapezoidal integration
    IWP = integrand.integrate('lev')

    IWP.attrs = {
        'long_name': 'ice water path',
        'units': 'kg m-2',
        'description': 'vertically integrated ice water'
    }

    return IWP


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'compute_relative_humidity',
    'compute_saturation_mixing_ratio',
    'compute_column_water_vapor',
    'compute_liquid_water_path',
    'compute_ice_water_path',
]
