"""
VVM Reader Manifest Management

This package provides functionality for creating, loading, and querying
variable manifests that describe which variables are available in which
output groups.
"""

# Manager class
from .manager import ManifestManager

# Creation functions
from .creation import (
    create_manifest_for_simulation,
    validate_manifest_format,
)

# Loading and query functions
from .operations import (
    load_manifest_if_available,
    load_or_create_manifest,
    get_groups_for_variables,
    get_variables_in_group,
    filter_variables_for_group,
    get_manifest_summary,
    print_manifest_info,
)

__all__ = [
    # Manager class
    "ManifestManager",
    # Creation functions
    "create_manifest_for_simulation",
    "validate_manifest_format",
    # Loading functions
    "load_manifest_if_available",
    "load_or_create_manifest",
    # Query functions
    "get_groups_for_variables",
    "get_variables_in_group",
    "filter_variables_for_group",
    # Analysis functions
    "get_manifest_summary",
    "print_manifest_info",
]
