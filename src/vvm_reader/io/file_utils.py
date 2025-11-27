"""
VVM Reader File Operation Utilities

This module handles all file and directory operations including path parsing,
file searching, date extraction, and directory validation.
"""

import re
from pathlib import Path
from typing import List, Optional, Sequence, Dict
from functools import lru_cache
import xarray as xr

from ..core.config import (
    OUTPUT_INDEX_PATTERN, GROUP_FILE_PATTERN, 
    get_simulation_paths, KNOWN_GROUPS
)
from ..core.exceptions import (
    InvalidFormatError, validate_simulation_directory, validate_required_file
)

# ============================================================================
# File Name Parsing
# ============================================================================

def parse_index_from_filename(path: Path) -> int:
    """
    Parse 6-digit index from VVM output filename.
    
    Examples:
        filename-000123.nc -> 123
        
    Args:
        path: File path
        
    Returns:
        int: Parsed file index
        
    Raises:
        InvalidFormatError: If index cannot be parsed
    """
    filename = path.name
    match = re.search(OUTPUT_INDEX_PATTERN, filename)
    if not match:
        raise InvalidFormatError("filename", "*-XXXXXX.nc", filename)
    
    try:
        return int(match.group(1))
    except ValueError as e:
        raise InvalidFormatError("file index", "6-digit integer", f"{match.group(1)} ({e})")

def extract_group_from_filename(filename: str) -> Optional[str]:
    """
    Extract group name from VVM output filename.
    
    Examples:
        sim_name.L.Dynamic-000001.nc -> L.Dynamic
        
    Args:
        filename: Filename to parse
        
    Returns:
        Optional[str]: Group name if found, None otherwise
    """
    match = re.match(r"[^.]+\.(.+)-\d{6}\.nc$", filename)
    return match.group(1) if match else None

# ============================================================================
# File and Directory Operations
# ============================================================================

@lru_cache(maxsize=32)
def list_group_files(archive_dir: Path, group: str) -> List[Path]:
    """
    List all files for a specific output group.
    
    Args:
        archive_dir: Archive directory path
        group: Group name (e.g., "L.Dynamic")
        
    Returns:
        List[Path]: Sorted list of files for the group
    """
    pattern = GROUP_FILE_PATTERN.format(group=group)
    return sorted(archive_dir.glob(pattern))

@lru_cache(maxsize=32)
def list_available_groups(sim_dir: Path) -> List[str]:
    """
    List all available output groups in a simulation.
    
    Args:
        sim_dir: Simulation directory path
        
    Returns:
        List[str]: Sorted list of available group names
    """
    try:
        archive_dir = validate_simulation_directory(sim_dir)
    except Exception:
        return []
    
    groups = set()
    for nc_file in archive_dir.glob("*.nc"):
        group_name = extract_group_from_filename(nc_file.name)
        if group_name:
            groups.add(group_name)
    
    return sorted(groups)

@lru_cache(maxsize=32)
def list_variables_in_group(sim_dir: Path, group: str, engine: Optional[str] = None) -> List[str]:
    """
    List all variables in a specific output group.
    
    Args:
        sim_dir: Simulation directory path
        group: Group name
        engine: xarray backend engine
        
    Returns:
        List[str]: Sorted list of variable names
    """
    archive_dir = validate_simulation_directory(sim_dir)
    files = list_group_files(archive_dir, group)
    
    if not files:
        return []
    
    try:
        with xr.open_dataset(files[0], engine=engine) as ds:
            return sorted(list(ds.data_vars))
    except Exception:
        return []

def filter_files_by_groups(archive_dir: Path, groups: Sequence[str]) -> Dict[str, List[Path]]:
    """
    Filter files by output groups.
    
    Args:
        archive_dir: Archive directory path
        groups: List of group names to include
        
    Returns:
        Dict[str, List[Path]]: Mapping of group names to file lists
    """
    result = {}
    for group in groups:
        files = list_group_files(archive_dir, group)
        if files:
            result[group] = files
    return result

# ============================================================================
# Group and Variable Selection
# ============================================================================

def resolve_groups_to_load(
    sim_dir: Path,
    requested_groups: Optional[Sequence[str]],
    variables: Optional[Sequence[str]],
    manifest: Optional[Dict] = None
) -> List[str]:
    """
    Determine which groups to load based on available groups, requested variables, and manifest.

    Handles variable name disambiguation using priority rules from config.

    Args:
        sim_dir: Simulation directory path
        requested_groups: Explicitly requested groups
        variables: Requested variables (used to infer groups from manifest)
        manifest: Variable manifest for group inference

    Returns:
        List[str]: Groups to load
    """
    available_groups = list_available_groups(sim_dir)

    # If groups explicitly requested, validate and return
    if requested_groups is not None:
        from ..core.exceptions import check_groups_availability
        check_groups_availability(requested_groups, available_groups)
        return list(requested_groups)

    # If variables requested and manifest available, infer groups
    if variables is not None and manifest is not None:
        from ..core.config import (
            VARIABLE_GROUP_PRIORITY, STAGGERED_VARIABLES,
            PREFER_3D_FOR_STAGGERED, VERTICAL_DIM
        )

        inferred_groups = []
        for var in variables:
            var_info = manifest.get(var, {})
            var_groups = var_info.get("groups", []) if isinstance(var_info, dict) else []

            if not var_groups:
                continue

            # Apply disambiguation logic
            selected_group = None

            # Priority 1: Check VARIABLE_GROUP_PRIORITY for explicit rules
            if var in VARIABLE_GROUP_PRIORITY:
                priority_list = VARIABLE_GROUP_PRIORITY[var]
                for priority_group in priority_list:
                    if priority_group in var_groups and priority_group in available_groups:
                        selected_group = priority_group
                        break

            # Priority 2: For staggered variables, prefer 3D groups (with 'lev' dimension)
            if selected_group is None and var in STAGGERED_VARIABLES and PREFER_3D_FOR_STAGGERED:
                for group in var_groups:
                    if group not in available_groups:
                        continue
                    # Check if this group's version has 'lev' dimension
                    var_dims = var_info.get("dims", [])
                    if VERTICAL_DIM in var_dims:
                        selected_group = group
                        break

            # Priority 3: Use first available group
            if selected_group is None:
                for group in var_groups:
                    if group in available_groups:
                        selected_group = group
                        break

            # Add selected group to list
            if selected_group and selected_group not in inferred_groups:
                inferred_groups.append(selected_group)

        if inferred_groups:
            return inferred_groups

    # Default: return all available known groups
    return [g for g in available_groups if g in KNOWN_GROUPS]

def get_variables_for_group(
    group: str, 
    requested_variables: Optional[Sequence[str]], 
    manifest: Optional[Dict] = None
) -> Optional[List[str]]:
    """
    Get the subset of variables to load for a specific group.
    
    Args:
        group: Group name
        requested_variables: All requested variables
        manifest: Variable manifest
        
    Returns:
        Optional[List[str]]: Variables to load for this group, or None for all variables
    """
    if requested_variables is None:
        return None
    
    if manifest is not None:
        # Use manifest to filter variables for this group
        group_vars = []
        for var in requested_variables:
            var_info = manifest.get(var, {})
            if isinstance(var_info, dict) and group in var_info.get("groups", []):
                group_vars.append(var)
        return group_vars if group_vars else None
    
    # Without manifest, we can't determine which variables belong to which group
    # Return all requested variables and let xarray filter
    return list(requested_variables)

# ============================================================================
# Simulation Information
# ============================================================================

def get_simulation_info(sim_dir: Path) -> Dict:
    """
    Get comprehensive information about a simulation directory.
    
    Args:
        sim_dir: Simulation directory path
        
    Returns:
        Dict: Simulation information including files, groups, dates, etc.
    """
    info = {
        'sim_dir': sim_dir,
        'sim_name': sim_dir.name,
        'valid': False,
        'has_archive': False,
        'has_topo': False,
        'has_fort98': False,
        'available_groups': [],
        'total_files': 0,
        'errors': []
    }
    
    try:
        # Validate simulation directory
        archive_dir = validate_simulation_directory(sim_dir)
        info['has_archive'] = True
        info['valid'] = True
        
        # Count files and list groups
        nc_files = list(archive_dir.glob("*.nc"))
        info['total_files'] = len(nc_files)
        info['available_groups'] = list_available_groups(sim_dir)
        
    except Exception as e:
        info['errors'].append(f"Directory validation: {e}")
    
    # Check for required files
    paths = get_simulation_paths(sim_dir)
    
    try:
        validate_required_file(paths['topo'], "TOPO.nc")
        info['has_topo'] = True
    except Exception as e:
        info['errors'].append(f"TOPO.nc: {e}")
    
    try:
        validate_required_file(paths['fort98'], "fort.98")
        info['has_fort98'] = True
    except Exception as e:
        info['errors'].append(f"fort.98: {e}")
    
    return info

def is_valid_simulation_dir(sim_dir: Path) -> bool:
    """
    Quick check if a directory is a valid VVM simulation.
    
    Args:
        sim_dir: Directory path to check
        
    Returns:
        bool: True if valid simulation directory
    """
    try:
        validate_simulation_directory(sim_dir)
        return True
    except Exception:
        return False

def find_simulation_directories(parent_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Find all valid simulation directories under a parent directory.
    
    Args:
        parent_dir: Parent directory to search
        recursive: Whether to search recursively
        
    Returns:
        List[Path]: List of valid simulation directories
    """
    sim_dirs = []
    
    search_paths = parent_dir.rglob("*") if recursive else parent_dir.iterdir()
    
    for path in search_paths:
        if path.is_dir() and is_valid_simulation_dir(path):
            sim_dirs.append(path)
    
    return sorted(sim_dirs)