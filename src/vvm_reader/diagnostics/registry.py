"""
Diagnostic Variable Registry System

This module provides a registry for diagnostic variables, tracking their:
- Calculation functions
- Dependencies (model output variables and reference profiles needed)
- Metadata (long_name, units, description)

The registry enables automatic dependency resolution and ensures diagnostic
variables are computed in the correct order.
"""

from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Registry Data Structures
# ============================================================================

@dataclass
class DiagnosticVariable:
    """
    Metadata and computation information for a diagnostic variable.

    Attributes:
        name: Variable name (e.g., 'T', 'RH', 'MSE')
        compute_func: Function to compute this variable
        file_dependencies: Model output variables needed from NetCDF files
        profile_dependencies: Reference profiles needed from fort.98
        diagnostic_dependencies: Other diagnostic variables needed
        long_name: Descriptive name
        units: Physical units
        description: Detailed description
        standard_name: CF convention standard name (if applicable)
    """
    name: str
    compute_func: Callable
    file_dependencies: Set[str] = field(default_factory=set)
    profile_dependencies: Set[str] = field(default_factory=set)
    diagnostic_dependencies: Set[str] = field(default_factory=set)
    long_name: str = ""
    units: str = ""
    description: str = ""
    standard_name: Optional[str] = None

    def __post_init__(self):
        """Convert lists to sets if necessary."""
        if isinstance(self.file_dependencies, (list, tuple)):
            self.file_dependencies = set(self.file_dependencies)
        if isinstance(self.profile_dependencies, (list, tuple)):
            self.profile_dependencies = set(self.profile_dependencies)
        if isinstance(self.diagnostic_dependencies, (list, tuple)):
            self.diagnostic_dependencies = set(self.diagnostic_dependencies)

    @property
    def all_dependencies(self) -> Set[str]:
        """Get all dependencies (file + profile + diagnostic)."""
        return (self.file_dependencies |
                self.profile_dependencies |
                self.diagnostic_dependencies)


class DiagnosticRegistry:
    """
    Registry for diagnostic variables.

    This registry maintains a catalog of all available diagnostic variables
    and enables dependency resolution and computation ordering.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._registry: Dict[str, DiagnosticVariable] = {}
        logger.debug("Initialized diagnostic variable registry")

    def register(
        self,
        name: str,
        compute_func: Callable,
        file_dependencies: Optional[List[str]] = None,
        profile_dependencies: Optional[List[str]] = None,
        diagnostic_dependencies: Optional[List[str]] = None,
        long_name: str = "",
        units: str = "",
        description: str = "",
        standard_name: Optional[str] = None,
    ) -> None:
        """
        Register a diagnostic variable.

        Args:
            name: Variable name
            compute_func: Function to compute the variable
            file_dependencies: Model output variables needed
            profile_dependencies: Reference profiles needed
            diagnostic_dependencies: Other diagnostic variables needed
            long_name: Descriptive name
            units: Physical units
            description: Detailed description
            standard_name: CF convention standard name

        Raises:
            ValueError: If variable already registered
        """
        if name in self._registry:
            logger.warning(f"Diagnostic variable '{name}' already registered, overwriting")

        diag_var = DiagnosticVariable(
            name=name,
            compute_func=compute_func,
            file_dependencies=set(file_dependencies or []),
            profile_dependencies=set(profile_dependencies or []),
            diagnostic_dependencies=set(diagnostic_dependencies or []),
            long_name=long_name,
            units=units,
            description=description,
            standard_name=standard_name,
        )

        self._registry[name] = diag_var
        logger.debug(f"Registered diagnostic variable: {name}")

    def is_registered(self, name: str) -> bool:
        """Check if a variable is registered."""
        return name in self._registry

    def get(self, name: str) -> Optional[DiagnosticVariable]:
        """Get diagnostic variable metadata."""
        return self._registry.get(name)

    def get_file_dependencies(self, variables: List[str]) -> Set[str]:
        """
        Get all file dependencies for a list of diagnostic variables.

        Args:
            variables: List of diagnostic variable names

        Returns:
            Set of file variable names needed

        Raises:
            KeyError: If any variable is not registered
        """
        file_deps = set()

        for var in variables:
            if var not in self._registry:
                raise KeyError(f"Diagnostic variable '{var}' not registered")

            diag_var = self._registry[var]
            file_deps.update(diag_var.file_dependencies)

            # Recursively get dependencies of diagnostic dependencies
            for dep in diag_var.diagnostic_dependencies:
                file_deps.update(self.get_file_dependencies([dep]))

        return file_deps

    def get_profile_dependencies(self, variables: List[str]) -> Set[str]:
        """
        Get all profile dependencies for a list of diagnostic variables.

        Args:
            variables: List of diagnostic variable names

        Returns:
            Set of profile names needed (e.g., 'PIBAR', 'RHO')

        Raises:
            KeyError: If any variable is not registered
        """
        profile_deps = set()

        for var in variables:
            if var not in self._registry:
                raise KeyError(f"Diagnostic variable '{var}' not registered")

            diag_var = self._registry[var]
            profile_deps.update(diag_var.profile_dependencies)

            # Recursively get dependencies of diagnostic dependencies
            for dep in diag_var.diagnostic_dependencies:
                profile_deps.update(self.get_profile_dependencies([dep]))

        return profile_deps

    def resolve_computation_order(self, variables: List[str]) -> List[str]:
        """
        Resolve the order in which diagnostic variables should be computed.

        Uses topological sorting to ensure dependencies are computed before
        variables that depend on them.

        Args:
            variables: List of diagnostic variable names to compute

        Returns:
            List of variable names in computation order

        Raises:
            KeyError: If any variable is not registered
            ValueError: If circular dependencies are detected
        """
        # Build dependency graph
        graph = {}
        in_degree = {}

        # Initialize all requested variables
        to_process = set(variables)
        processed = set()

        while to_process:
            var = to_process.pop()
            if var in processed:
                continue

            if var not in self._registry:
                raise KeyError(f"Diagnostic variable '{var}' not registered")

            diag_var = self._registry[var]
            graph[var] = diag_var.diagnostic_dependencies.copy()

            # Add diagnostic dependencies to processing queue
            to_process.update(diag_var.diagnostic_dependencies)
            processed.add(var)

        # Compute in-degrees for Kahn's algorithm
        # in_degree[v] = number of edges pointing TO v
        # In dependency graph: in_degree[v] = number of variables that depend ON v
        all_vars = set(graph.keys())
        in_degree = {var: len(graph[var]) for var in all_vars}

        # Topological sort (Kahn's algorithm)
        # Start with nodes that have no dependencies (in_degree = 0)
        queue = [var for var in all_vars if in_degree[var] == 0]
        result = []

        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            var = queue.pop(0)
            result.append(var)

            # For each variable that depends on 'var'
            for dependent in all_vars:
                if var in graph[dependent]:
                    # Remove this dependency edge
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for circular dependencies
        if len(result) != len(all_vars):
            remaining = all_vars - set(result)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        # Result is already in correct order (dependencies first)
        # No need to reverse - Kahn's algorithm produces the correct topological order
        logger.debug(f"Resolved computation order: {result}")
        return result

    def list_all(self) -> List[str]:
        """List all registered diagnostic variables."""
        return sorted(self._registry.keys())

    def get_metadata(self, name: str) -> Dict[str, str]:
        """
        Get metadata for a diagnostic variable.

        Args:
            name: Variable name

        Returns:
            Dictionary with metadata (long_name, units, etc.)
        """
        if name not in self._registry:
            raise KeyError(f"Diagnostic variable '{name}' not registered")

        diag_var = self._registry[name]
        return {
            'long_name': diag_var.long_name,
            'units': diag_var.units,
            'description': diag_var.description,
            'standard_name': diag_var.standard_name or '',
        }

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"DiagnosticRegistry({len(self._registry)} variables registered)"


# ============================================================================
# Global Registry Instance
# ============================================================================

# Global registry instance (populated by diagnostic modules)
_global_registry = DiagnosticRegistry()


def get_registry() -> DiagnosticRegistry:
    """Get the global diagnostic variable registry."""
    return _global_registry


def register_diagnostic(
    name: str,
    file_dependencies: Optional[List[str]] = None,
    profile_dependencies: Optional[List[str]] = None,
    diagnostic_dependencies: Optional[List[str]] = None,
    long_name: str = "",
    units: str = "",
    description: str = "",
    standard_name: Optional[str] = None,
):
    """
    Decorator to register a diagnostic variable computation function.

    Example:
        @register_diagnostic(
            name='T',
            file_dependencies=['th'],
            profile_dependencies=['PIBAR'],
            long_name='Temperature',
            units='K'
        )
        def compute_temperature(ds, profiles, diagnostics):
            # profiles is xr.Dataset with lev coordinate
            # xarray will automatically align coordinates
            return ds['th'] * profiles['PIBAR']

    Args:
        name: Variable name
        file_dependencies: Model output variables needed
        profile_dependencies: Reference profiles needed
        diagnostic_dependencies: Other diagnostic variables needed
        long_name: Descriptive name
        units: Physical units
        description: Detailed description
        standard_name: CF convention standard name

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        _global_registry.register(
            name=name,
            compute_func=func,
            file_dependencies=file_dependencies,
            profile_dependencies=profile_dependencies,
            diagnostic_dependencies=diagnostic_dependencies,
            long_name=long_name,
            units=units,
            description=description,
            standard_name=standard_name,
        )
        return func

    return decorator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DiagnosticVariable',
    'DiagnosticRegistry',
    'get_registry',
    'register_diagnostic',
]
