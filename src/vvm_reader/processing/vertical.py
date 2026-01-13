"""
VVM Reader Vertical Level Processing

This module handles vertical coordinate processing including level selection and fort.98 reading.
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xarray as xr

from ..core.config import VERTICAL_DIM, FORT98_FILENAME, FORT98_SEARCH_LINES
from ..core.core_types import VerticalSelection
from ..core.exceptions import RequiredFileNotFoundError, ParameterError, DataProcessingError

# ============================================================================
# fort.98 Processing
# ============================================================================

def read_reference_profiles_from_fort98(sim_dir: Path) -> dict:
    """
    Read reference state profiles (RHO, THBAR, PBAR, PIBAR, QVBAR) from fort.98.

    The fort.98 file contains initial state profiles with the format:
    - Header line: "K, RHO(K),THBAR(K),PBAR(K),PIBAR(K),QVBAR(K)"
    - Separator line: "="
    - Data lines: K  RHO  THBAR  PBAR  PIBAR  QVBAR

    Returns profiles for K=1..(Kmax-1), excluding the topmost level
    to match VVM output dimensions.

    Args:
        sim_dir: Simulation directory path

    Returns:
        dict: Dictionary with keys 'RHO', 'THBAR', 'PBAR', 'PIBAR', 'QVBAR'
              Each value is a numpy array of shape (nz,)

    Raises:
        RequiredFileNotFoundError: If fort.98 file doesn't exist
        DataProcessingError: If file format is invalid

    """
    fort98_path = sim_dir / FORT98_FILENAME

    if not fort98_path.is_file():
        raise RequiredFileNotFoundError(fort98_path, "fort.98")

    try:
        with open(fort98_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise DataProcessingError("fort.98 reading", f"Cannot read file: {e}")

    # Find the reference profiles header line
    header_line_idx = None
    target_header = "K, RHO(K),THBAR(K),PBAR(K),PIBAR(K),QVBAR(K)"

    for i, line in enumerate(lines[:FORT98_SEARCH_LINES]):
        line_stripped = ' '.join(line.split())
        if target_header in line_stripped:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise DataProcessingError(
            "fort.98 parsing",
            f"Reference profiles header '{target_header}' not found in first {FORT98_SEARCH_LINES} lines"
        )

    # Parse data lines (skip header and separator line)
    rho_values = []
    thbar_values = []
    pbar_values = []
    pibar_values = []
    qvbar_values = []
    data_start_line = header_line_idx + 2

    for i in range(data_start_line, len(lines)):
        line = lines[i].strip()

        # Stop at empty line or another separator/header
        if not line or line.startswith('=') or 'K,' in line:
            break

        try:
            tokens = line.split()
            if len(tokens) < 6:
                break

            # Parse: K  RHO  THBAR  PBAR  PIBAR  QVBAR
            rho_values.append(float(tokens[1]))
            thbar_values.append(float(tokens[2]))
            pbar_values.append(float(tokens[3]))
            pibar_values.append(float(tokens[4]))
            qvbar_values.append(float(tokens[5]))
        except (ValueError, IndexError) as e:
            raise DataProcessingError(
                "fort.98 parsing",
                f"Invalid reference profile data at line {i+1}: '{line}' - {e}"
            )

    if not rho_values:
        raise DataProcessingError("fort.98 parsing", "No valid reference profile values found")

    # Remove the topmost level (not used in model output)
    if len(rho_values) > 1:
        rho_values = rho_values[:-1]
        thbar_values = thbar_values[:-1]
        pbar_values = pbar_values[:-1]
        pibar_values = pibar_values[:-1]
        qvbar_values = qvbar_values[:-1]

    return {
        'RHO': np.array(rho_values, dtype=np.float64),
        'THBAR': np.array(thbar_values, dtype=np.float64),
        'PBAR': np.array(pbar_values, dtype=np.float64),
        'PIBAR': np.array(pibar_values, dtype=np.float64),
        'QVBAR': np.array(qvbar_values, dtype=np.float64),
    }


def read_vertical_levels_from_fort98(sim_dir: Path) -> np.ndarray:
    """
    Read ZT(K) vertical levels from fort.98 file with caching.

    The fort.98 file contains vertical level information with the format:
    - Header line: "K, ZZ(K),ZT(K),FNZ(K),FNT(K)"
    - Separator line: "="
    - Data lines: K  ZZ(K)  ZT(K)  FNZ(K)  FNT(K)

    Returns ZT(K) values for K=1..(Kmax-1), excluding the topmost level.

    Args:
        sim_dir: Simulation directory path

    Returns:
        np.ndarray: ZT(K) levels in meters

    Raises:
        RequiredFileNotFoundError: If fort.98 file doesn't exist
        DataProcessingError: If file format is invalid

    """
    fort98_path = sim_dir / FORT98_FILENAME
    
    if not fort98_path.is_file():
        raise RequiredFileNotFoundError(fort98_path, "fort.98")
    
    try:
        with open(fort98_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise DataProcessingError("fort.98 reading", f"Cannot read file: {e}")
    
    # Find the exact header line
    header_line_idx = None
    target_header = "K, ZZ(K),ZT(K),FNZ(K),FNT(K)"
    
    for i, line in enumerate(lines[:FORT98_SEARCH_LINES]):
        # Clean up whitespace and compare
        line_stripped = ' '.join(line.split())  # Normalize whitespace
        if target_header in line_stripped:
            header_line_idx = i
            break
    
    if header_line_idx is None:
        raise DataProcessingError(
            "fort.98 parsing", 
            f"Header line '{target_header}' not found in first {FORT98_SEARCH_LINES} lines"
        )
    
    # Parse data lines (skip header and separator line)
    zt_values = []
    data_start_line = header_line_idx + 2  # Skip header and separator
    
    for i in range(data_start_line, len(lines)):
        line = lines[i].strip()
        
        # Stop at empty line or another separator/header
        if not line or line.startswith('=') or 'K,' in line:
            break
        
        try:
            tokens = line.split()
            if len(tokens) < 3:
                break
            zt_value = float(tokens[2])  # ZT(K) is the 3rd column (0-indexed: 2)
            zt_values.append(zt_value)
        except (ValueError, IndexError) as e:
            raise DataProcessingError(
                "fort.98 parsing", 
                f"Invalid data format at line {i+1}: '{line}' - {e}"
            )
    
    if not zt_values:
        raise DataProcessingError("fort.98 parsing", "No valid ZT(K) values found")
    
    # Remove the topmost level (not used in model output)
    if len(zt_values) > 1:
        zt_values = zt_values[:-1]
    
    return np.array(zt_values, dtype=np.float64)

# ============================================================================
# Vertical Selection Types and Helpers
# ============================================================================

@dataclass
class VerticalSelectionInfo:
    """
    Information about resolved vertical selection.
    
    Attributes:
        read_selection: Slice or array for reading data (may include extra levels for centering)
        final_positions: Positions of final indices within expanded array (for arbitrary selection)
        is_contiguous: Whether selection is contiguous (slice) or arbitrary (array)
        original_indices: Original requested indices (for arbitrary selection)
    """
    read_selection: Union[slice, np.ndarray, None]
    final_positions: Optional[np.ndarray] = None
    is_contiguous: bool = True
    original_indices: Optional[np.ndarray] = None


def resolve_heights_to_nearest_indices(
    heights: np.ndarray,
    levels: np.ndarray
) -> np.ndarray:
    """
    Convert heights to nearest level indices.
    
    Args:
        heights: Array of target heights in meters
        levels: Array of available vertical levels in meters
        
    Returns:
        Array of nearest level indices
    """
    indices = np.array([np.argmin(np.abs(levels - h)) for h in heights])
    return np.unique(indices)  # Remove duplicates and sort


def expand_indices_for_centering(
    indices: np.ndarray,
    needs_z_centering: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand indices to include neighbors needed for z-staggered centering.
    
    Maintains boundary behavior consistent with slice-based centering:
    - Only adds k-1 when k > 0 (same as extend_vertical_slice_for_centering)
    - Does not add k+1 (_safe_shift will produce NaN at boundaries)
    
    Args:
        indices: Original level indices
        needs_z_centering: Whether z-direction centering (w, eta, xi) is needed
        
    Returns:
        Tuple of (expanded_indices, final_positions):
            - expanded_indices: All indices to read (including neighbors)
            - final_positions: Positions of original indices in expanded array
    """
    if not needs_z_centering:
        return indices, np.arange(len(indices))
    
    expanded = set(indices)
    for k in indices:
        # Consistent with extend_vertical_slice_for_centering: only add k-1 when k > 0
        if k > 0:
            expanded.add(k - 1)
    
    expanded_indices = np.array(sorted(expanded))
    final_positions = np.searchsorted(expanded_indices, indices)
    
    return expanded_indices, final_positions


def filter_to_final_indices(
    dataset: xr.Dataset,
    final_positions: np.ndarray
) -> xr.Dataset:
    """
    Extract final level indices from expanded dataset after centering.
    
    Args:
        dataset: Dataset with expanded vertical levels
        final_positions: Positions of final indices in the expanded array
        
    Returns:
        Dataset with only the originally requested levels
    """
    if VERTICAL_DIM not in dataset.dims:
        return dataset
    
    return dataset.isel({VERTICAL_DIM: final_positions})


# ============================================================================
# Vertical Slice Resolution
# ============================================================================

def resolve_vertical_slice(
    sim_dir: Path,
    vertical_selection: VerticalSelection
) -> Optional[slice]:
    """
    Resolve vertical slice based on index or height range.
    
    Uses index-based selection when available (faster), falls back to height-based selection.
    
    Args:
        sim_dir: Simulation directory path  
        vertical_selection: Vertical selection parameters
        
    Returns:
        Optional[slice]: Vertical slice, or None if no constraints
        
    Raises:
        RequiredFileNotFoundError: If fort.98 needed but not found
        DataProcessingError: If height conversion fails
    """
    # Index-based selection takes priority (faster)
    if vertical_selection.uses_index_selection:
        k0, k1 = vertical_selection.index_range
        k0 = max(0, int(k0))
        k1 = int(k1)
        return slice(k0, k1 + 1) if k1 >= k0 else None
    
    # Height-based selection (requires fort.98 reading)
    elif vertical_selection.uses_height_selection:
        # This will now raise proper exceptions instead of returning None
        levels = read_vertical_levels_from_fort98(sim_dir)
        
        zmin, zmax = vertical_selection.height_range
        z_min, z_max = min(float(zmin), float(zmax)), max(float(zmin), float(zmax))
        
        # Find indices where levels fall within the height range
        indices = np.where((levels >= z_min) & (levels <= z_max))[0]
        
        if indices.size > 0:
            return slice(int(indices.min()), int(indices.max()) + 1)
        else:
            raise DataProcessingError(
                "height range conversion",
                f"No vertical levels found in height range [{z_min}, {z_max}]"
            )
    
    return None


def resolve_vertical_selection(
    sim_dir: Path,
    vertical_selection: VerticalSelection,
    needs_z_centering: bool = False
) -> VerticalSelectionInfo:
    """
    Resolve vertical selection for all modes (contiguous and arbitrary).
    
    Handles both slice-based (contiguous) and array-based (arbitrary) selection.
    For arbitrary selection with z-centering needed, automatically expands indices
    to include neighbors required for centering.
    
    Priority: level_indices > heights > index_range > height_range
    
    Args:
        sim_dir: Simulation directory path
        vertical_selection: Vertical selection parameters
        needs_z_centering: Whether z-direction centering (w, eta, xi) is needed
        
    Returns:
        VerticalSelectionInfo with read_selection, final_positions, and is_contiguous flag
        
    Raises:
        RequiredFileNotFoundError: If fort.98 needed but not found
        DataProcessingError: If conversion fails
    """
    # Priority 1: Arbitrary indices (level_indices)
    if vertical_selection.uses_arbitrary_indices:
        indices = np.array(sorted(set(vertical_selection.level_indices)))
        expanded, final_positions = expand_indices_for_centering(indices, needs_z_centering)
        return VerticalSelectionInfo(
            read_selection=expanded,
            final_positions=final_positions,
            is_contiguous=False,
            original_indices=indices
        )
    
    # Priority 2: Arbitrary heights
    if vertical_selection.uses_arbitrary_heights:
        levels = read_vertical_levels_from_fort98(sim_dir)
        heights = np.array(vertical_selection.heights)
        indices = resolve_heights_to_nearest_indices(heights, levels)
        expanded, final_positions = expand_indices_for_centering(indices, needs_z_centering)
        return VerticalSelectionInfo(
            read_selection=expanded,
            final_positions=final_positions,
            is_contiguous=False,
            original_indices=indices
        )
    
    # Priority 3 & 4: Contiguous selection (index_range or height_range)
    # Use existing resolve_vertical_slice for backward compatibility
    vertical_slice = resolve_vertical_slice(sim_dir, vertical_selection)
    return VerticalSelectionInfo(
        read_selection=vertical_slice,
        final_positions=None,
        is_contiguous=True,
        original_indices=None
    )

def extend_vertical_slice_for_centering(
    vertical_slice: Optional[slice],
    needs_w_centering: bool = False
) -> Tuple[Optional[slice], int, Optional[int]]:
    """
    Extend vertical slice to include halo for w-wind centering.
    
    Args:
        vertical_slice: Original vertical slice
        needs_w_centering: Whether w-wind centering is needed
        
    Returns:
        Tuple[Optional[slice], int, Optional[int]]: Extended slice, crop offset, target length
    """
    if not needs_w_centering or vertical_slice is None:
        return vertical_slice, 0, None
    
    k0 = vertical_slice.start or 0
    k1 = vertical_slice.stop
    target_length = k1 - k0
    
    # Add one level below if possible
    if k0 > 0:
        extended_slice = slice(k0 - 1, k1)
        crop_offset = 1
        return extended_slice, crop_offset, target_length
    
    return vertical_slice, 0, target_length

# ============================================================================
# Surface Extraction
# ============================================================================

def extract_surface_nearest_values(
    dataset: xr.Dataset,
    surface_level: xr.DataArray,
    vertical_selection: VerticalSelection
) -> xr.Dataset:
    """
    Extract surface-nearest values for 3D variables.

    For each horizontal grid point, extracts the value at vertical level = topo + 1
    (the first level above terrain).

    Args:
        dataset: Input dataset
        surface_level: surface-nearest level indices
        vertical_selection: Vertical selection parameters

    Returns:
        xr.Dataset: Dataset with surface values added or replaced

    Note:
        When surface_only=True, the surface_level indices are saved as
        'surface_level_index' in the dataset for diagnostic variable computation.
    """
    if not vertical_selection.surface_nearest or VERTICAL_DIM not in dataset.sizes:
        return dataset

    try:
        # Convert topo to integer indices
        sfc_indices = surface_level.astype(np.int64)
        
        # Ensure indices are computed (numpy array) before using for indexing
        # This is critical for performance with dask arrays and to avoid "vindex does not support dask" errors
        if hasattr(sfc_indices, "compute"):
            sfc_indices = sfc_indices.compute()

        # Identify variables that have the vertical dimension
        vars_with_vertical = [
            name for name, var in dataset.data_vars.items() 
            if VERTICAL_DIM in var.dims
        ]
        
        if not vars_with_vertical:
            return dataset

        # Optimization 1: Filter variables only
        ds_subset = dataset[vars_with_vertical]

        # Optimization 2: Vectorized extraction
        # Extract values at sfc_indices for all 3D variables at once using advanced indexing
        surface_ds = ds_subset.isel({VERTICAL_DIM: sfc_indices})

        # The extracted surface_ds might still contain the vertical coordinate (e.g., 'lev') 
        # but as a 2D variable (since it was indexed). 
        # We drop it to avoid conflicts with the main dataset's 1D vertical coordinate.
        if VERTICAL_DIM in surface_ds.coords:
            surface_ds = surface_ds.drop_vars(VERTICAL_DIM)

        if vertical_selection.surface_only:
            # Strategy: 
            # 1. Remove all variables using vertical dimension from the original dataset
            # 2. Merge the new surface variables
            
            # Drop the vertical dimension and all associated variables
            new_ds = dataset.drop_dims(VERTICAL_DIM)
            
            # Restore attributes for the surface variables
            for name in surface_ds.data_vars:
                if name in dataset:
                    surface_ds[name].attrs = dataset[name].attrs
                    surface_ds[name].encoding = getattr(dataset[name], 'encoding', {})

            # Merge surface variables - skip checks for speed as we know coords align
            new_ds = new_ds.merge(surface_ds, compat='override', join='override')

            # Save surface level indices for diagnostic computation
            sfc_indices_var = sfc_indices.copy()
            # If sfc_indices was computed to numpy, wrap it back to DataArray for the dataset
            if not isinstance(sfc_indices_var, (xr.DataArray, xr.Variable)):
                 sfc_indices_var = xr.DataArray(sfc_indices, dims=surface_level.dims, coords=surface_level.coords)

            sfc_indices_var.attrs = {
                'long_name': 'surface level index',
                'description': 'Vertical level index used for surface extraction (k = topo + 1)',
                'units': 'level'
            }
            new_ds['surface_level_index'] = sfc_indices_var

            return new_ds
        
        else:
            # Strategy:
            # 1. Rename surface variables with suffix
            # 2. Merge into original dataset
            
            rename_map = {}
            for name in surface_ds.data_vars:
                new_name = f"{name}{vertical_selection.surface_suffix}"
                rename_map[name] = new_name
                
            surface_ds = surface_ds.rename(rename_map)
            
            # Restore/Update attributes
            for old_name, new_name in rename_map.items():
                if old_name in dataset:
                    surface_ds[new_name].attrs = dataset[old_name].attrs
                    surface_ds[new_name].encoding = getattr(dataset[old_name], 'encoding', {})
            
            # Skip checks for speed as we know coords align
            return dataset.merge(surface_ds, compat='override', join='override')
    
    except Exception as e:
        raise DataProcessingError("surface extraction", str(e))

# ============================================================================
# Vertical Coordinate Utilities
# ============================================================================

def validate_vertical_selection(
    vertical_selection: VerticalSelection,
    available_levels: Optional[int] = None
) -> None:
    """
    Validate vertical selection parameters.
    
    Args:
        vertical_selection: Vertical selection parameters
        available_levels: Number of available vertical levels
        
    Raises:
        ParameterError: If parameters are invalid
    """
    if vertical_selection.index_range is not None:
        k0, k1 = vertical_selection.index_range
        
        if k0 < 0:
            raise ParameterError("index_range", str(vertical_selection.index_range), 
                               "Start index must be non-negative")
        
        if available_levels is not None and k1 >= available_levels:
            raise ParameterError("index_range", str(vertical_selection.index_range),
                               f"End index must be < {available_levels}")
    
    if vertical_selection.height_range is not None:
        z0, z1 = vertical_selection.height_range
        if z0 < 0 or z1 < 0:
            raise ParameterError("height_range", str(vertical_selection.height_range),
                               "Heights must be non-negative")

def apply_vertical_selection(
    dataset: xr.Dataset,
    vertical_slice: Optional[slice]
) -> xr.Dataset:
    """
    Apply vertical slice to dataset.
    
    Args:
        dataset: Input dataset
        vertical_slice: Vertical slice to apply
        
    Returns:
        xr.Dataset: Dataset with vertical selection applied
    """
    if vertical_slice is None or VERTICAL_DIM not in dataset.sizes:
        return dataset
    
    return dataset.isel({VERTICAL_DIM: vertical_slice})

def ensure_vertical_coordinate_in_meters(dataset: xr.Dataset) -> xr.Dataset:
    """
    Ensure the vertical coordinate uses meters as its unit.

    Converts `lev` from kilometers to meters when coordinate metadata
    indicates kilometer units, updating the units attribute accordingly.

    Args:
        dataset: Dataset containing the vertical coordinate.

    Returns:
        xr.Dataset: Dataset with meter-based vertical coordinate.
    """
    if VERTICAL_DIM not in dataset.coords:
        return dataset

    lev_coord = dataset[VERTICAL_DIM]
    units = str(lev_coord.attrs.get('units', '')).lower() if hasattr(lev_coord, 'attrs') else ''

    if units in {'m', 'meter', 'meters'}:
        return dataset

    if units in {'km', 'kilometer', 'kilometers', 'level'}:
        converted = lev_coord * 1000.0
        converted.attrs = dict(lev_coord.attrs) if hasattr(lev_coord, 'attrs') else {}
        converted.attrs['units'] = 'm'
        encoding = getattr(lev_coord, 'encoding', None)
        if encoding:
            converted.encoding = dict(encoding)
        return dataset.assign_coords({VERTICAL_DIM: converted})

    return dataset


def crop_vertical_after_centering(
    dataset: xr.Dataset,
    crop_offset: int,
    target_length: Optional[int]
) -> xr.Dataset:
    """
    Crop vertical dimension after wind centering operations.
    
    Args:
        dataset: Dataset after centering
        crop_offset: Number of levels to crop from bottom
        target_length: Target length of vertical dimension
        
    Returns:
        xr.Dataset: Cropped dataset
    """
    if (crop_offset == 0 or target_length is None or 
        VERTICAL_DIM not in dataset.sizes):
        return dataset
    
    vertical_slice = slice(crop_offset, crop_offset + target_length)
    return dataset.isel({VERTICAL_DIM: vertical_slice})
