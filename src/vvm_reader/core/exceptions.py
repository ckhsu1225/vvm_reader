"""
VVM Reader Custom Exception Classes

This module defines all custom exception classes for better error handling
and more informative error messages.
"""

import json
from typing import Optional, Sequence
from pathlib import Path

# ============================================================================
# Base Exception
# ============================================================================

class VVMReaderError(Exception):
    """Base exception class for all VVM Reader related errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        full_message = f"{message}\nDetails: {details}" if details else message
        super().__init__(full_message)

# ============================================================================
# File and Directory Errors
# ============================================================================

class SimulationDirectoryError(VVMReaderError):
    """Simulation directory related errors."""
    
    def __init__(self, sim_dir: Path, reason: str):
        super().__init__(f"Invalid simulation directory: {sim_dir}", reason)
        self.sim_dir = sim_dir

class ArchiveNotFoundError(VVMReaderError):
    """Archive directory not found."""
    
    def __init__(self, archive_dir: Path):
        super().__init__(f"Archive directory not found: {archive_dir}")
        self.archive_dir = archive_dir

class RequiredFileNotFoundError(VVMReaderError):
    """Required file not found."""
    
    def __init__(self, file_path: Path, file_type: str):
        super().__init__(f"{file_type} file not found: {file_path}")
        self.file_path = file_path
        self.file_type = file_type

# ============================================================================
# Data Format Errors
# ============================================================================

class InvalidFormatError(VVMReaderError):
    """Invalid data format errors."""
    
    def __init__(self, item: str, expected_format: str, actual: str):
        super().__init__(
            f"Invalid format for {item}: {actual}",
            f"Expected format: {expected_format}"
        )
        self.item = item
        self.expected_format = expected_format
        self.actual = actual

class CoordinateError(VVMReaderError):
    """Coordinate system related errors."""
    
    def __init__(self, coord_name: str, issue: str):
        super().__init__(f"Coordinate error in '{coord_name}': {issue}")
        self.coord_name = coord_name

# ============================================================================
# Data Availability Errors
# ============================================================================

class GroupNotFoundError(VVMReaderError):
    """Output groups not found."""
    
    def __init__(self, missing_groups: Sequence[str], available_groups: Optional[Sequence[str]] = None):
        groups_str = ", ".join(missing_groups)
        super().__init__(
            f"Output groups not found: {groups_str}",
            f"Available groups: {', '.join(available_groups)}" if available_groups else None
        )
        self.missing_groups = list(missing_groups)
        self.available_groups = list(available_groups) if available_groups else None

class VariableNotFoundError(VVMReaderError):
    """Variables not found."""
    
    def __init__(self, missing_variables: Sequence[str], available_variables: Optional[Sequence[str]] = None):
        vars_str = ", ".join(missing_variables)
        super().__init__(
            f"Variables not found: {vars_str}",
            f"Available variables: {', '.join(sorted(available_variables))}" if available_variables else None
        )
        self.missing_variables = list(missing_variables)
        self.available_variables = list(available_variables) if available_variables else None

class NoDataError(VVMReaderError):
    """No data files found matching criteria."""
    
    def __init__(self, criteria: str):
        super().__init__(f"No data files found matching criteria: {criteria}")
        self.criteria = criteria

# ============================================================================
# Manifest Errors
# ============================================================================

class ManifestError(VVMReaderError):
    """
    Base class for variable manifest related errors.
    
    Provides common functionality for all manifest-related errors.
    """
    
    def __init__(self, message: str, manifest_path: Optional[Path] = None, details: Optional[str] = None):
        if manifest_path:
            message = f"{message} (manifest: {manifest_path})"
        super().__init__(message, details)
        self.manifest_path = manifest_path

class ManifestIOError(ManifestError):
    """Manifest file I/O errors."""
    
    def __init__(self, operation: str, path: Path, reason: str):
        message = f"Failed to {operation} manifest"
        super().__init__(message, path, reason)
        self.operation = operation

class ManifestFormatError(ManifestError):
    """Invalid manifest format errors."""
    
    def __init__(self, issue: str, path: Optional[Path] = None):
        message = f"Invalid manifest format: {issue}"
        super().__init__(message, path)
        self.issue = issue

class ManifestNotFoundError(ManifestError):
    """Manifest file not found error."""
    
    def __init__(self, path: Path):
        message = f"Variable manifest not found"
        super().__init__(message, path)

# Convenience function for manifest error handling
def handle_manifest_error(operation: str, path: Path, error: Exception) -> None:
    """
    Handle manifest-related errors and raise appropriate exceptions.
    
    Args:
        operation: Operation being performed ("load", "save", "create")
        path: Manifest file path
        error: Original exception
        
    Raises:
        ManifestNotFoundError: If file not found
        ManifestFormatError: If JSON parsing fails
        ManifestIOError: For other I/O errors
    """
    if isinstance(error, FileNotFoundError):
        raise ManifestNotFoundError(path)
    elif isinstance(error, (json.JSONDecodeError, ValueError)) and "JSON" in str(error):
        raise ManifestFormatError(str(error), path)
    else:
        raise ManifestIOError(operation, path, str(error))

# ============================================================================
# Processing Errors
# ============================================================================

class DataProcessingError(VVMReaderError):
    """Data processing related errors."""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(f"Data processing failed during {operation}", reason)
        self.operation = operation

class ParameterError(VVMReaderError):
    """Parameter validation errors."""
    
    def __init__(self, parameter: str, value: str, reason: str):
        super().__init__(f"Invalid parameter '{parameter}': {value}", reason)
        self.parameter = parameter
        self.value = value

# ============================================================================
# Utility Functions
# ============================================================================

def validate_simulation_directory(sim_dir: Path) -> Path:
    """
    Validate simulation directory and return archive path.
    
    Args:
        sim_dir: Simulation directory path
        
    Returns:
        Path: Archive directory path
        
    Raises:
        SimulationDirectoryError: If directory is invalid
        ArchiveNotFoundError: If archive directory not found
    """
    archive_dir = sim_dir / "archive"
    if not archive_dir.is_dir():
        raise ArchiveNotFoundError(archive_dir)
    
    return archive_dir

def validate_required_file(file_path: Path, file_type: str) -> Path:
    """
    Validate that a required file exists.
    
    Args:
        file_path: Path to the file
        file_type: Type description of the file
        
    Returns:
        Path: The validated file path
        
    Raises:
        RequiredFileNotFoundError: If file doesn't exist
    """
    if not file_path.is_file():
        raise RequiredFileNotFoundError(file_path, file_type)
    return file_path

def check_groups_availability(requested: Sequence[str], available: Sequence[str]) -> None:
    """Check if all requested groups are available."""
    missing = [g for g in requested if g not in available]
    if missing:
        raise GroupNotFoundError(missing, available)

def check_variables_availability(requested: Sequence[str], available: Sequence[str]) -> None:
    """Check if all requested variables are available."""
    missing = [v for v in requested if v not in available]
    if missing:
        raise VariableNotFoundError(missing, available)