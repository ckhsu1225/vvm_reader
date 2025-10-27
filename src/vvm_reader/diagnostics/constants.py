"""
Physical Constants for Atmospheric Diagnostics

This module defines physical constants used in diagnostic variable calculations.
All constants follow SI units unless otherwise specified.

References:
    - WMO recommended values
    - Wallace & Hobbs (2006) Atmospheric Science: An Introductory Survey
"""

# ============================================================================
# Fundamental Physical Constants
# ============================================================================

# Gas constants
R_d = 287.05          # Specific gas constant for dry air [J kg^-1 K^-1]
R_v = 461.5           # Specific gas constant for water vapor [J kg^-1 K^-1]
epsilon = R_d / R_v   # Ratio of gas constants ≈ 0.622 [dimensionless]

# Specific heats (at constant pressure)
Cp_d = 1004.7         # Specific heat of dry air [J kg^-1 K^-1]
Cp_v = 1860.1         # Specific heat of water vapor [J kg^-1 K^-1]
Cp_l = 4219.4         # Specific heat of liquid water [J kg^-1 K^-1]
Cp_i = 2090.0         # Specific heat of ice [J kg^-1 K^-1]

# Specific heats (at constant volume)
Cv_d = 717.6          # Specific heat of dry air at constant volume [J kg^-1 K^-1]

# Ratio of specific heats
kappa = R_d / Cp_d    # ≈ 0.286 [dimensionless]

# Latent heats (at 0°C)
Lv = 2.501e6          # Latent heat of vaporization [J kg^-1]
Lf = 3.337e5          # Latent heat of fusion [J kg^-1]
Ls = Lv + Lf          # Latent heat of sublimation [J kg^-1]

# ============================================================================
# Reference Values
# ============================================================================

P0 = 1000.0e2         # Reference pressure [Pa] = 1000 hPa
T0 = 273.15           # Reference temperature (0°C) [K]
T_triple = 273.16     # Triple point of water [K]

# ============================================================================
# Earth Constants
# ============================================================================

g = 9.80665           # Gravitational acceleration [m s^-2]
omega = 7.292e-5      # Earth's angular velocity [rad s^-1]
a_earth = 6.371e6     # Earth's mean radius [m]

# ============================================================================
# Saturation Vapor Pressure Parameters
# ============================================================================

# Parameters for saturation vapor pressure calculation
# Using improved August-Roche-Magnus formula (Alduchov & Eskridge, 1996)

# Over liquid water (valid for -40°C to 50°C)
es0_liquid = 611.2    # Reference saturation vapor pressure [Pa]
a_liquid = 17.67      # Empirical constant [dimensionless]
b_liquid = 243.5      # Empirical constant [K]

# Over ice (valid for -80°C to 0°C)
es0_ice = 611.2       # Reference saturation vapor pressure [Pa]
a_ice = 22.46         # Empirical constant [dimensionless]
b_ice = 272.62        # Empirical constant [K]

# Temperature thresholds for phase transitions
T_freeze = 273.15     # Freezing point of water [K] = 0°C
T_ice_threshold = 250.15  # Mixed-phase lower threshold [K] = -23°C
# Note: T_ice_threshold follows ECMWF and many NWP models' convention.
# Physical basis: At T < -23°C, supercooled liquid droplets are rare in natural clouds.
# Alternative values used in literature:
#   - Strict: 233.15 K (-40°C, homogeneous freezing temperature)
#   - Conservative: 253.15 K (-20°C, significant ice nucleation)
#   - Simple: 273.15 K (0°C, no mixed phase)
# References:
#   - Pruppacher & Klett (1997): Microphysics of Clouds and Precipitation
#   - ECMWF IFS Documentation (Cy47r3, 2021)
#   - Rogers & Yau (1989): A Short Course in Cloud Physics

# ============================================================================
# Derived Constants
# ============================================================================

# Virtual temperature constants
epsilon_inv = 1.0 / epsilon  # ≈ 1.608

# Density constants (at reference conditions)
rho_liquid = 999.97   # Density of liquid water [kg m^-3]
rho_ice = 917.0       # Density of ice [kg m^-3]

# ============================================================================
# Utility Functions for Common Conversions
# ============================================================================

def celsius_to_kelvin(T_celsius):
    """Convert temperature from Celsius to Kelvin."""
    return T_celsius + T0

def kelvin_to_celsius(T_kelvin):
    """Convert temperature from Kelvin to Celsius."""
    return T_kelvin - T0

def hPa_to_Pa(p_hPa):
    """Convert pressure from hPa to Pa."""
    return p_hPa * 100.0

def Pa_to_hPa(p_Pa):
    """Convert pressure from Pa to hPa."""
    return p_Pa / 100.0

# ============================================================================
# Physical Constant Validation
# ============================================================================

def validate_constants():
    """
    Validate that physical constants are self-consistent.

    Returns:
        bool: True if all validation checks pass

    Raises:
        AssertionError: If any consistency check fails
    """
    # Check gas constant ratio
    assert abs(epsilon - 0.622) < 0.001, "epsilon should be approximately 0.622"

    # Check kappa
    assert abs(kappa - 0.286) < 0.001, "kappa should be approximately 0.286"

    # Check sublimation = vaporization + fusion
    assert abs((Lv + Lf) - Ls) < 1.0, "Ls should equal Lv + Lf"

    # Check reference pressure
    assert abs(P0 - 1.0e5) < 1.0, "P0 should be 100000 Pa (1000 hPa)"

    return True

# Run validation on import (optional, can be commented out for production)
# validate_constants()

# ============================================================================
# Package Metadata
# ============================================================================

__all__ = [
    # Gas constants
    'R_d', 'R_v', 'epsilon', 'epsilon_inv',

    # Specific heats
    'Cp_d', 'Cp_v', 'Cp_l', 'Cp_i', 'Cv_d', 'kappa',

    # Latent heats
    'Lv', 'Lf', 'Ls',

    # Reference values
    'P0', 'T0', 'T_triple', 'T_freeze',

    # Earth constants
    'g', 'omega', 'a_earth',

    # Saturation vapor pressure parameters
    'es0_liquid', 'a_liquid', 'b_liquid',
    'es0_ice', 'a_ice', 'b_ice', 'T_ice_threshold',

    # Derived constants
    'rho_liquid', 'rho_ice',

    # Utility functions
    'celsius_to_kelvin', 'kelvin_to_celsius',
    'hPa_to_Pa', 'Pa_to_hPa',
    'validate_constants',
]
