"""
VVM Reader Manifest Operations

This module provides functions for loading, querying, and analyzing
variable manifests.
"""

from pathlib import Path
from typing import Optional, Union, Mapping, List, Dict, Any

from ...core.core_types import VariableManifest
from .manager import ManifestManager
from .creation import create_manifest_for_simulation


# ============================================================================
# Manifest Loading Utilities
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


# ============================================================================
# Manifest Query Functions
# ============================================================================

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


# ============================================================================
# Manifest Analysis Functions
# ============================================================================

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
