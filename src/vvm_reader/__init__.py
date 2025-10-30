"""
VVM Reader - A Python package for reading VVM model output data.

This package provides efficient and flexible tools for loading and processing
VVM (Vector Vorticity equation cloud-resolving Model) output data into 
xarray datasets.

Key Features:
- Structured parameter interface for better organization
- Automatic terrain masking and wind centering
- Flexible spatial, temporal, and vertical selection
- Efficient chunked loading with Dask
- Variable manifests for optimized group selection
- Backward compatibility with legacy interfaces

Quick Start:
    >>> import vvm_reader as vvm
    >>> ds = vvm.open_vvm_dataset("/path/to/simulation")
    >>> 
    >>> # Load specific variables with regional selection
    >>> region = vvm.Region(lon_range=(120, 122), lat_range=(23, 25))
    >>> ds = vvm.open_vvm_dataset(
    ...     "/path/to/simulation",
    ...     variables=["u", "v", "w", "th"],
    ...     region=region
    ... )
"""

__version__ = "2.0.0"
__author__ = "VVM Reader Development Team"

# Import main interface functions
from .main import (
    # Primary interface (recommended)
    open_vvm_dataset,
    
    # Convenience functions
    quick_load,
    load_surface_data,
    load_region,
    load_indices,
    
    # Utility functions
    list_available_simulations,
    get_simulation_info,
    get_coordinate_info,
    get_terrain_info,
    get_vertical_info,
    get_spatial_info,
    get_terrain_height,
    get_reference_profiles,
    convert_coordinates_to_indices,
    convert_indices_to_coordinates,
    convert_time_to_indices,
    convert_heights_to_indices,
    create_variable_manifest,
)

# Import parameter classes for structured interface
from .core.core_types import (
    Region,
    TimeSelection,
    VerticalSelection, 
    ProcessingOptions,
    LoadParameters,
)

# Import configuration for advanced users
from .core.config import (
    KNOWN_GROUPS,
    WIND_VARIABLES,
    VORTICITY_VARIABLES,
    STAGGERED_VARIABLES,
    STAGGER_CONFIG,
    VARIABLE_GROUP_PRIORITY,
    VERTICAL_DIM,
    TIME_DIM,
)

# Import exceptions for error handling
from .core.exceptions import (
    VVMReaderError,
    SimulationDirectoryError,
    GroupNotFoundError,
    VariableNotFoundError,
    ManifestError,
    DataProcessingError,
)

# Import logging configuration
from .core.logging_config import setup_logging, set_log_level

# Import diagnostics module
from . import diagnostics
from .diagnostics import (
    compute_diagnostics,
    list_available_diagnostics,
)

# Define what gets imported with "from vvm_reader import *"
__all__ = [
    # Version info
    '__version__',
    
    # Main interface functions
    'open_vvm_dataset',
    'quick_load',
    'load_surface_data',
    'load_region',
    'load_indices',
    
    # Utility functions
    'list_available_simulations',
    'get_simulation_info',
    'get_coordinate_info',
    'get_terrain_info',
    'get_vertical_info',
    'get_spatial_info',
    'get_terrain_height',
    'get_reference_profiles',
    'convert_coordinates_to_indices',
    'convert_indices_to_coordinates',
    'convert_time_to_indices',
    'convert_heights_to_indices',
    'create_variable_manifest',
    
    # Parameter classes
    'Region',
    'TimeSelection',
    'VerticalSelection',
    'ProcessingOptions',
    'LoadParameters',
    
    # Configuration constants
    'KNOWN_GROUPS',
    'WIND_VARIABLES',
    'VORTICITY_VARIABLES',
    'STAGGERED_VARIABLES',
    'STAGGER_CONFIG',
    'VARIABLE_GROUP_PRIORITY',
    'VERTICAL_DIM',
    'TIME_DIM',
    
    # Exception classes
    'VVMReaderError',
    'SimulationDirectoryError',
    'GroupNotFoundError',
    'VariableNotFoundError',
    'ManifestError',
    'DataProcessingError',

    # Logging configuration
    'setup_logging',
    'set_log_level',

    # Diagnostics
    'diagnostics',
    'compute_diagnostics',
    'list_available_diagnostics',
]

# Package metadata
__doc_format__ = "restructuredtext"

def print_package_info():
    """Print package information and usage examples."""
    print(f"""
VVM Reader v{__version__}
========================

A Python package for reading VVM model output data.

Quick Examples:
--------------

1. Load all data from a simulation:
   >>> import vvm_reader as vvm
   >>> ds = vvm.open_vvm_dataset("/path/to/simulation")

2. Load specific variables:
   >>> ds = vvm.open_vvm_dataset(
   ...     "/path/to/simulation",
   ...     variables=["u", "v", "w", "th"]
   ... )

3. Regional selection:
   >>> region = vvm.Region(lon_range=(120, 122), lat_range=(23, 25))
   >>> ds = vvm.open_vvm_dataset("/path/to/simulation", region=region)

4. Time selection:
   >>> time_sel = vvm.TimeSelection(time_index_range=(0, 36))
   >>> ds = vvm.open_vvm_dataset("/path/to/simulation", time_selection=time_sel)

5. Surface data only:
   >>> ds = vvm.load_surface_data("/path/to/simulation", ["u", "v", "th"])

6. Quick load (first N time steps):
   >>> ds = vvm.quick_load("/path/to/simulation", ["u", "v", "w"], time_steps=10)

For more information, see the documentation or use help(vvm.open_vvm_dataset).
""")

# Optional: Set up logging
import logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# Optional: Check for optional dependencies and warn if missing
def _check_optional_dependencies():
    """Check for optional dependencies and warn if missing."""
    try:
        import dask
    except ImportError:
        _logger.warning("Dask not available. Large datasets may cause memory issues.")
    
    try:
        import netCDF4
    except ImportError:
        _logger.warning("netCDF4 not available. Some file formats may not be supported.")

# Run dependency check on import (optional)
# _check_optional_dependencies()