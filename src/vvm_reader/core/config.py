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
# Staggered Grid Variables
# ============================================================================

# Wind variables on staggered grid
WIND_VARIABLES = ("u", "v", "w")

# Vorticity variables on staggered grid
VORTICITY_VARIABLES = ("zeta", "eta", "xi")

# All variables requiring centering from staggered grid to cell center
STAGGERED_VARIABLES = WIND_VARIABLES + VORTICITY_VARIABLES

# Stagger configuration: which dimensions each variable is staggered in
# theta is at cell center (i, j, k) - reference point
STAGGER_CONFIG = {
    "u": ("x",),        # staggered in x: at (i+0.5, j, k)
    "v": ("y",),        # staggered in y: at (i, j+0.5, k)
    "w": ("z",),        # staggered in z: at (i, j, k+0.5)
    "zeta": ("x", "y"), # staggered in x,y: at (i+0.5, j+0.5, k)
    "eta": ("x", "z"),  # staggered in x,z: at (i+0.5, j, k+0.5)
    "xi": ("y", "z"),   # staggered in y,z: at (i, j+0.5, k+0.5)
}

# ============================================================================
# Variable Name Disambiguation
# ============================================================================

# Some variable names appear in multiple groups with different meanings.
# This dictionary specifies group priority for ambiguous variable names.
# Higher priority groups are checked first when loading variables.
VARIABLE_GROUP_PRIORITY = {
    "eta": ["L.Dynamic", "C.LandSurface"],  # eta in L.Dynamic is vorticity (3D)
                                             # eta in C.LandSurface is latent heat flux (2D)
}

# For staggered variables, prefer groups that contain 3D data (with 'lev' dimension)
# This helps disambiguate cases where variable names conflict between 2D and 3D versions
PREFER_3D_FOR_STAGGERED = True

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