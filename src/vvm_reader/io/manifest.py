"""
VVM Reader Variable Manifest Management

This module handles creation, loading, and management of variable manifests
that describe which variables are available in which output groups.
"""

import json
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, Union, Mapping, List

from ..core.config import get_default_manifest_path
from ..core.core_types import VariableManifest
from ..core.exceptions import ManifestIOError
from ..io.file_utils import list_available_groups, list_group_files

# ============================================================================
# Manifest Management Class
# ============================================================================

class ManifestManager:
    """
    Manager class for variable manifests.
    
    Handles creation, loading, saving, and querying of variable manifests
    that describe which variables are available in which output groups.
    """
    
    def __init__(self, manifest_path: Optional[Path] = None):
        """
        Initialize manifest manager.
        
        Args:
            manifest_path: Custom path for manifest file, uses default if None
        """
        self.manifest_path = manifest_path or get_default_manifest_path()
    
    def create_manifest(
        self,
        sim_dir: Path,
        engine: Optional[str] = None,
        include_coords: bool = False,
        save_manifest: bool = True
    ) -> VariableManifest:
        """
        Create a comprehensive variable manifest for a simulation.
        
        The manifest maps each variable to:
        - groups: List of output groups containing this variable
        - attrs: Variable attributes (units, long_name, etc.)
        - dims: Variable dimensions
        - dtype: Data type
        
        Args:
            sim_dir: Simulation directory path
            engine: xarray backend engine
            include_coords: Whether to include coordinate variables
            save_manifest: Whether to save manifest to file
            
        Returns:
            VariableManifest: Created manifest dictionary
        """
        from ..io.file_utils import validate_simulation_directory
        
        archive_dir = validate_simulation_directory(sim_dir)
        groups = list_available_groups(sim_dir)
        
        variable_map: Dict[str, Any] = {}
        
        for group in groups:
            files = list_group_files(archive_dir, group)
            if not files:
                continue
            
            try:
                with xr.open_dataset(files[0], engine=engine) as ds:
                    # Get variable names
                    if include_coords:
                        var_names = list(ds.variables.keys())
                    else:
                        var_names = list(ds.data_vars.keys())
                    
                    for var_name in var_names:
                        self._process_variable(variable_map, var_name, group, ds)
                        
            except Exception as e:
                # Log error but continue with other groups
                print(f"Warning: Failed to process group {group}: {e}")
                continue
        
        if save_manifest:
            self.save_manifest(variable_map)
        
        return variable_map
    
    def _process_variable(
        self,
        variable_map: Dict[str, Any],
        var_name: str,
        group: str,
        dataset: xr.Dataset
    ) -> None:
        """Process a single variable for the manifest."""
        # Initialize or get existing entry
        if var_name not in variable_map:
            variable_map[var_name] = {
                "groups": [],
                "attrs": {},
                "dims": [],
                "dtype": None
            }
        
        entry = variable_map[var_name]
        
        # Add group if not already present
        if group not in entry["groups"]:
            entry["groups"].append(group)
        
        try:
            var_data = dataset[var_name]
            
            # Extract variable attributes
            attrs = self._extract_variable_attributes(var_data)
            
            # Merge attributes (prefer non-empty values)
            for key, value in attrs.items():
                if key not in entry["attrs"] or not entry["attrs"][key]:
                    entry["attrs"][key] = value
            
            # Set dimensions (use first occurrence)
            if not entry["dims"]:
                entry["dims"] = list(var_data.sizes)
            
            # Set data type (use first occurrence)
            if entry["dtype"] is None:
                try:
                    entry["dtype"] = str(var_data.dtype)
                except Exception:
                    pass
                    
        except Exception:
            # If variable processing fails, at least keep the group mapping
            pass
    
    def _extract_variable_attributes(self, var_data: xr.DataArray) -> Dict[str, Any]:
        """Extract relevant attributes from a variable."""
        attrs = {}
        
        # Standard attributes to extract
        attr_keys = ["long_name", "standard_name", "units", "description"]
        
        for key in attr_keys:
            if key in var_data.attrs:
                value = var_data.attrs[key]
                # Only keep JSON-serializable values
                if isinstance(value, (str, int, float, bool)):
                    attrs[key] = value
        
        return attrs
    
    def load_manifest(self, path: Optional[Path] = None) -> VariableManifest:
        """
        Load variable manifest from JSON file.
        
        Args:
            path: Custom manifest path, uses default if None
            
        Returns:
            VariableManifest: Loaded manifest dictionary
            
        Raises:
            ManifestIOError: If loading fails
        """
        manifest_path = path or self.manifest_path
        
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ManifestIOError("load", manifest_path, "File not found")
        except json.JSONDecodeError as e:
            raise ManifestIOError("load", manifest_path, f"Invalid JSON: {e}")
        except Exception as e:
            raise ManifestIOError("load", manifest_path, str(e))
    
    def save_manifest(
        self,
        manifest: VariableManifest,
        path: Optional[Path] = None
    ) -> None:
        """
        Save variable manifest to JSON file.
        
        Args:
            manifest: Manifest dictionary to save
            path: Custom save path, uses default if None
            
        Raises:
            ManifestIOError: If saving fails
        """
        manifest_path = path or self.manifest_path
        
        try:
            # Ensure directory exists
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                
        except Exception as e:
            raise ManifestIOError("save", manifest_path, str(e))
    
    def manifest_exists(self, path: Optional[Path] = None) -> bool:
        """Check if manifest file exists."""
        manifest_path = path or self.manifest_path
        return manifest_path.is_file()

# ============================================================================
# Manifest Utility Functions
# ============================================================================

def load_manifest_if_available(
    manifest_spec: Optional[Union[str, Path, Mapping[str, Any]]]
) -> Optional[VariableManifest]:
    """
    Load a variable manifest if available.
    
    Args:
        manifest_spec: Manifest specification (path, dict, or None)
        
    Returns:
        Optional[VariableManifest]: Loaded manifest or None if not available
    """
    if manifest_spec is None:
        # Try to load default manifest
        try:
            manager = ManifestManager()
            return manager.load_manifest()
        except Exception:
            return None
    
    if isinstance(manifest_spec, (str, Path)):
        # Load from specified path
        try:
            manager = ManifestManager(Path(manifest_spec))
            return manager.load_manifest()
        except Exception:
            return None
    
    if isinstance(manifest_spec, Mapping):
        # Convert mapping to proper manifest format
        return dict(manifest_spec)
    
    return None

def load_or_create_manifest(
    manifest_spec: Optional[Union[str, Path, Mapping[str, Any]]],
    sim_dir: Optional[Path] = None
) -> Optional[VariableManifest]:
    """
    Load a variable manifest, creating one if needed and sim_dir is provided.
    
    Args:
        manifest_spec: Manifest specification (path, dict, or None)
        sim_dir: Simulation directory for manifest creation if needed
        
    Returns:
        Optional[VariableManifest]: Loaded/created manifest or None
    """
    # First try to load existing manifest
    manifest = load_manifest_if_available(manifest_spec)
    if manifest is not None:
        return manifest
    
    # If loading failed and we have a sim_dir, try to create manifest
    if sim_dir is not None:
        try:
            return create_manifest_for_simulation(sim_dir, save_manifest=False)
        except Exception:
            return None
    
    return None

def get_groups_for_variables(
    variables: List[str],
    manifest: VariableManifest
) -> List[str]:
    """
    Get the minimal set of groups needed for specified variables.
    
    Args:
        variables: List of variable names
        manifest: Variable manifest
        
    Returns:
        List[str]: Minimal list of groups containing the variables
    """
    required_groups = []
    
    for var in variables:
        var_info = manifest.get(var, {})
        if isinstance(var_info, dict):
            var_groups = var_info.get("groups", [])
            for group in var_groups:
                if group not in required_groups:
                    required_groups.append(group)
    
    return required_groups

def get_variables_in_group(
    group: str,
    manifest: VariableManifest
) -> List[str]:
    """
    Get all variables available in a specific group.
    
    Args:
        group: Group name
        manifest: Variable manifest
        
    Returns:
        List[str]: List of variables in the group
    """
    variables = []
    
    for var_name, var_info in manifest.items():
        if isinstance(var_info, dict):
            var_groups = var_info.get("groups", [])
            if group in var_groups:
                variables.append(var_name)
    
    return sorted(variables)

def filter_variables_for_group(
    variables: List[str],
    group: str,
    manifest: VariableManifest
) -> List[str]:
    """
    Filter variables to only those available in a specific group.
    
    Args:
        variables: List of variable names to filter
        group: Group name
        manifest: Variable manifest
        
    Returns:
        List[str]: Filtered list of variables available in the group
    """
    filtered = []
    
    for var in variables:
        var_info = manifest.get(var, {})
        if isinstance(var_info, dict):
            var_groups = var_info.get("groups", [])
            if group in var_groups:
                filtered.append(var)
    
    return filtered

def get_manifest_summary(manifest: VariableManifest) -> Dict[str, Any]:
    """
    Get summary statistics about a manifest.
    
    Args:
        manifest: Variable manifest
        
    Returns:
        Dict[str, Any]: Summary information
    """
    if not manifest:
        return {
            "total_variables": 0,
            "total_groups": 0,
            "groups": [],
            "variables_per_group": {},
            "groups_per_variable": {}
        }
    
    all_groups = set()
    variables_per_group = {}
    groups_per_variable = {}
    
    for var_name, var_info in manifest.items():
        if not isinstance(var_info, dict):
            continue
            
        var_groups = var_info.get("groups", [])
        groups_per_variable[var_name] = len(var_groups)
        
        for group in var_groups:
            all_groups.add(group)
            if group not in variables_per_group:
                variables_per_group[group] = 0
            variables_per_group[group] += 1
    
    return {
        "total_variables": len(manifest),
        "total_groups": len(all_groups),
        "groups": sorted(all_groups),
        "variables_per_group": variables_per_group,
        "groups_per_variable": groups_per_variable
    }

def validate_manifest_format(manifest: Any) -> bool:
    """
    Validate that a manifest has the correct format.
    
    Args:
        manifest: Object to validate
        
    Returns:
        bool: True if format is valid
    """
    if not isinstance(manifest, dict):
        return False
    
    for var_name, var_info in manifest.items():
        if not isinstance(var_name, str):
            return False
        
        if not isinstance(var_info, dict):
            return False
        
        # Check required fields
        if "groups" not in var_info:
            return False
        
        if not isinstance(var_info["groups"], list):
            return False
        
        # Check optional fields have correct types
        if "attrs" in var_info and not isinstance(var_info["attrs"], dict):
            return False
        
        if "dims" in var_info and not isinstance(var_info["dims"], list):
            return False
        
        if "dtype" in var_info and var_info["dtype"] is not None:
            if not isinstance(var_info["dtype"], str):
                return False
    
    return True

# ============================================================================
# Manifest Creation Utilities
# ============================================================================

def create_manifest_for_simulation(
    sim_dir: Path,
    output_path: Optional[Path] = None,
    engine: Optional[str] = None,
    include_coords: bool = False,
    save_manifest: bool = True
) -> VariableManifest:
    """
    Create and save a manifest for a single simulation.
    
    Args:
        sim_dir: Simulation directory path
        output_path: Where to save manifest (default location if None)
        engine: xarray backend engine
        include_coords: Whether to include coordinate variables
        save_manifest: Whether to persist the manifest to disk
        
    Returns:
        VariableManifest: Created manifest
    """
    manager = ManifestManager(output_path)
    return manager.create_manifest(
        sim_dir=sim_dir,
        engine=engine,
        include_coords=include_coords,
        save_manifest=save_manifest
    )

def print_manifest_info(manifest: VariableManifest) -> None:
    """
    Print human-readable information about a manifest.
    
    Args:
        manifest: Variable manifest to describe
    """
    summary = get_manifest_summary(manifest)
    
    print(f"Variable Manifest Summary:")
    print(f"  Total variables: {summary['total_variables']}")
    print(f"  Total groups: {summary['total_groups']}")
    
    if summary['groups']:
        print(f"  Groups:")
        for group in summary['groups']:
            var_count = summary['variables_per_group'].get(group, 0)
            print(f"    {group}: {var_count} variables")
    
    # Show some example variables
    if manifest:
        print(f"  Example variables:")
        for i, (var_name, var_info) in enumerate(manifest.items()):
            if i >= 5:  # Show only first 5
                print(f"    ... and {len(manifest) - 5} more")
                break
            
            if isinstance(var_info, dict):
                groups = var_info.get("groups", [])
                units = var_info.get("attrs", {}).get("units", "")
                units_str = f" ({units})" if units else ""
                print(f"    {var_name}{units_str}: {groups}")