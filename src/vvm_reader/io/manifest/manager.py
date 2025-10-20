"""
VVM Reader Manifest Manager

This module provides the ManifestManager class for creating, loading,
and saving variable manifests.
"""

import json
import logging
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional

from ...core.config import get_default_manifest_path
from ...core.core_types import VariableManifest
from ...core.exceptions import ManifestIOError
from ..file_utils import list_available_groups, list_group_files

# Get logger for this module
logger = logging.getLogger('vvm_reader.io.manifest.manager')


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
        from ...core.exceptions import validate_simulation_directory

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
                logger.warning("Failed to process group %s: %s", group, e)
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
