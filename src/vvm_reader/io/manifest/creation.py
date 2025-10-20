"""
VVM Reader Manifest Creation

This module provides functions for creating and validating variable manifests.
"""

from pathlib import Path
from typing import Optional, Any

from ...core.core_types import VariableManifest
from .manager import ManifestManager


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
