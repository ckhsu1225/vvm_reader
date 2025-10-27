"""
VVM Diagnostics Module

This module provides diagnostic variable calculations for VVM model output.

Key Features:
- Automatic dependency resolution
- Background Exner function (PIBAR) based calculations
- Comprehensive thermodynamic, moisture, and energy diagnostics

Usage:
    import vvm_reader as vvm

    # Automatic diagnostic calculation
    ds = vvm.open_vvm_dataset(
        "/path/to/sim",
        variables=["u", "v", "t", "rh", "hms"]  # t, rh, hms computed automatically
    )

Available Diagnostic Variables:
    Thermodynamics:
        - t: Temperature
        - tv: Virtual temperature
        - thv: Virtual potential temperature
        - the: Equivalent potential temperature
        - thes: Saturation equivalent potential temperature

    Moisture:
        - rh: Relative humidity
        - qvs: Saturation mixing ratio
        - cwv: Column water vapor
        - lwp: Liquid water path
        - iwp: Ice water path

    Energy:
        - sd: Dry static energy
        - hm: Moist static energy
        - hms: Saturation moist static energy

Notes:
    - All calculations use PIBAR as background Exner function
    - Typical errors are <2% for most applications
"""

# Import all diagnostic calculation modules to register variables
from . import thermodynamics
from . import moisture
from . import energetics

# Import core components
from .constants import *
from .registry import get_registry, register_diagnostic

# Import computation engine
from .compute import (
    compute_diagnostics,
    list_available_diagnostics,
    get_diagnostic_metadata,
    get_required_file_variables,
    get_required_profiles,
    separate_file_and_diagnostic_variables,
)

__version__ = "1.0.0"

__all__ = [
    # Main computation functions
    'compute_diagnostics',
    'list_available_diagnostics',
    'get_diagnostic_metadata',
    'get_required_file_variables',
    'get_required_profiles',
    'separate_file_and_diagnostic_variables',

    # Registry
    'get_registry',
    'register_diagnostic',

    # Constants (from constants module)
    'R_d', 'R_v', 'Cp_d', 'Lv', 'g', 'P0', 'T0',
]


def print_diagnostic_info():
    """Print information about available diagnostic variables."""
    registry = get_registry()
    all_vars = registry.list_all()

    print(f"""
VVM Diagnostics v{__version__}
{'=' * 60}

Available Diagnostic Variables ({len(all_vars)}):
{'=' * 60}
""")

    # Group by category
    categories = {
        'Thermodynamics': ['t', 'tv', 'thv', 'the', 'thes'],
        'Moisture': ['rh', 'qvs', 'cwv', 'lwp', 'iwp'],
        'Energy': ['sd', 'hm', 'hms'],
    }

    for category, var_list in categories.items():
        print(f"\n{category}:")
        print("-" * 60)
        for var in var_list:
            if var in all_vars:
                metadata = registry.get_metadata(var)
                print(f"  {var:10s} - {metadata['long_name']:40s} [{metadata['units']}]")

    print("\n" + "=" * 60)
    print("\nFor more information, see documentation or use:")
    print("  >>> from vvm_reader.diagnostics import get_registry")
    print("  >>> registry = get_registry()")
    print("  >>> metadata = registry.get_metadata('t')")
    print()
