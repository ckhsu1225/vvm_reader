"""
VVM Reader Configuration and Constants

This module centralizes all configuration parameters, constants, and default values
for better maintainability and consistency across the codebase.
"""

import os
from pathlib import Path

# ============================================================================
# VVM Output Groups
# ============================================================================

KNOWN_GROUPS = (
    "C.LandSurface",
    "C.Surface",
    "L.Dynamic",
    "L.Radiation",
    "L.Thermodynamic",
    "L.Tracer",
)

# ============================================================================
# File Paths and Directories
# ============================================================================

# Use relative path for portability (relative to this module's location)
# Users can override via VVM_READER_DATA_DIR environment variable
DEFAULT_MANIFEST_DIR = Path(
    os.environ.get(
        "VVM_READER_DATA_DIR",
        str(Path(__file__).parent.parent / "data")
    )
)
DEFAULT_MANIFEST_FILENAME = "variable_manifest.json"

# ============================================================================
# Dimension Names
# ============================================================================

VERTICAL_DIM = 'lev'
TIME_DIM = 'time'
LON_DIM = 'lon'
LAT_DIM = 'lat'
TOPO_VAR = 'topo'

# ============================================================================
# File Patterns
# ============================================================================

OUTPUT_INDEX_PATTERN = r"-(\d{6})\.nc$"
DATE_PATTERN = r"(\d{8})"
GROUP_FILE_PATTERN = "*.{group}-*.nc"

# ============================================================================
# Default Processing Parameters
# ============================================================================

DEFAULT_CENTER_SUFFIX = "_c"
DEFAULT_SURFACE_SUFFIX = "_sfc"
DEFAULT_CHUNKS = "auto"

# ============================================================================
# Wind Variables (fixed lowercase)
# ============================================================================

WIND_VARIABLES = ("u", "v", "w")

# ============================================================================
# fort.98 Processing
# ============================================================================

FORT98_FILENAME = "fort.98"
FORT98_SEARCH_LINES = 500

# ============================================================================
# Terrain Processing
# ============================================================================

OCEAN_TOPO_VALUE = 0
MIN_LAND_TOPO_VALUE = 1

# ============================================================================
# Coordinate Processing
# ============================================================================

DATETIME_PRECISION = "ns"

# ============================================================================
# Helper Functions
# ============================================================================

def get_default_manifest_path() -> Path:
    """Get the default variable manifest path."""
    DEFAULT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_MANIFEST_DIR / DEFAULT_MANIFEST_FILENAME

def get_simulation_paths(sim_dir: Path) -> dict:
    """Get standard file paths for a simulation directory."""
    return {
        'archive': sim_dir / "archive",
        'topo': sim_dir / "TOPO.nc",
        'fort98': sim_dir / FORT98_FILENAME
    }