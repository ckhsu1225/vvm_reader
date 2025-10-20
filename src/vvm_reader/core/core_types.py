"""
VVM Reader Type Definitions and Data Classes

This module defines all data structures and type aliases used throughout the codebase
for better type safety and code clarity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any, Sequence, Mapping
from datetime import datetime
from pathlib import Path
import numpy as np

# ============================================================================
# Type Aliases
# ============================================================================

TimeValue = Union[str, datetime, np.datetime64]
TimeRange = Tuple[TimeValue, TimeValue]
IndexRange = Tuple[int, int]
CoordinateRange = Tuple[float, float]
ChunkSetting = Optional[Union[str, Dict[str, int]]]
VariableManifest = Dict[str, Any]

# ============================================================================
# Validation Utilities (Module Level)
# ============================================================================

def _validate_coordinate_range(name: str, range_val: Optional[CoordinateRange]) -> None:
    """Validate a coordinate range (lon/lat/height)."""
    if range_val is not None:
        if len(range_val) != 2:
            raise ValueError(f"{name} must contain exactly 2 values")
        if range_val[0] > range_val[1]:
            raise ValueError(f"{name}[0] must be <= {name}[1]")

def _validate_index_range(name: str, range_val: Optional[IndexRange], allow_negative: bool = False) -> None:
    """
    Validate an index range.
    
    Args:
        name: Parameter name for error messages
        range_val: Range to validate
        allow_negative: Whether to allow negative indices
    """
    if range_val is not None:
        if len(range_val) != 2:
            raise ValueError(f"{name} must contain exactly 2 values")
        if not allow_negative and range_val[0] < 0:
            raise ValueError(f"{name}[0] must be non-negative")
        if range_val[0] > range_val[1]:
            raise ValueError(f"{name}[0] must be <= {name}[1]")

# ============================================================================
# Spatial Region
# ============================================================================

@dataclass
class Region:
    """
    Spatial region definition for data selection.
    
    Supports both coordinate-based and index-based selection. When both are 
    specified, index-based selection takes priority.
    
    Attributes:
        lon_range: Longitude range (min_lon, max_lon) in degrees
        lat_range: Latitude range (min_lat, max_lat) in degrees
        x_range: X-index range (min_x, max_x) - takes priority over lon_range
        y_range: Y-index range (min_y, max_y) - takes priority over lat_range
    """
    lon_range: Optional[CoordinateRange] = None
    lat_range: Optional[CoordinateRange] = None
    x_range: Optional[IndexRange] = None
    y_range: Optional[IndexRange] = None
    
    def __post_init__(self):
        """Validate region parameters."""
        _validate_coordinate_range("lon_range", self.lon_range)
        _validate_coordinate_range("lat_range", self.lat_range)
        _validate_index_range("x_range", self.x_range)
        _validate_index_range("y_range", self.y_range)
        
        # Warn if both coordinate and index ranges are specified
        if self.lon_range is not None and self.x_range is not None:
            import warnings
            warnings.warn("Both lon_range and x_range specified. Using x_range (index-based).")
        
        if self.lat_range is not None and self.y_range is not None:
            import warnings
            warnings.warn("Both lat_range and y_range specified. Using y_range (index-based).")
    
    @property
    def has_selection(self) -> bool:
        """Check if any spatial selection is defined."""
        return (self.lon_range is not None or self.lat_range is not None or
                self.x_range is not None or self.y_range is not None)
    
    @property
    def uses_index_selection(self) -> bool:
        """Check if index-based selection is being used."""
        return self.x_range is not None or self.y_range is not None
    
    @property
    def uses_coordinate_selection(self) -> bool:
        """Check if coordinate-based selection is being used."""
        return self.lon_range is not None or self.lat_range is not None
    
    def get_effective_x_selection(self) -> Union[IndexRange, CoordinateRange, None]:
        """Get the effective x-direction selection (index takes priority)."""
        return self.x_range if self.x_range is not None else self.lon_range
    
    def get_effective_y_selection(self) -> Union[IndexRange, CoordinateRange, None]:
        """Get the effective y-direction selection (index takes priority)."""
        return self.y_range if self.y_range is not None else self.lat_range

# ============================================================================
# Time Selection
# ============================================================================

@dataclass 
class TimeSelection:
    """
    Time selection parameters for data loading.
    
    Supports both time-based and index-based selection. When both are specified,
    index-based selection takes priority for better performance.
    
    Attributes:
        time_range: Time range (start_time, end_time)
        time_index_range: File index range (start_index, end_index) - takes priority
    """
    time_range: Optional[TimeRange] = None
    time_index_range: Optional[IndexRange] = None
    
    def __post_init__(self):
        """Validate time selection parameters."""
        _validate_index_range("time_index_range", self.time_index_range)
        
        # Warn if both time and index ranges are specified
        if self.time_range is not None and self.time_index_range is not None:
            import warnings
            warnings.warn("Both time_range and time_index_range specified. Using time_index_range (index-based).")
    
    @property
    def has_selection(self) -> bool:
        """Check if any time selection is defined."""
        return self.time_range is not None or self.time_index_range is not None
    
    @property
    def uses_index_selection(self) -> bool:
        """Check if index-based selection is being used."""
        return self.time_index_range is not None
    
    @property
    def uses_time_selection(self) -> bool:
        """Check if time-based selection is being used."""
        return self.time_range is not None
    
    def get_effective_selection(self) -> Union[IndexRange, TimeRange, None]:
        """Get the effective time selection (index takes priority)."""
        return self.time_index_range if self.time_index_range is not None else self.time_range

# ============================================================================
# Vertical Selection
# ============================================================================

@dataclass
class VerticalSelection:
    """
    Vertical level selection and processing parameters.
    
    Supports both height-based and index-based selection. When both are specified,
    index-based selection takes priority for better performance.
    
    Attributes:
        index_range: Vertical index range (start_level, end_level) - takes priority
        height_range: Height range (min_height, max_height) in meters
        surface_nearest: Extract surface-nearest values
        surface_only: Keep only surface values (remove vertical dimension)
        surface_suffix: Suffix for surface variables
    """
    index_range: Optional[IndexRange] = None
    height_range: Optional[CoordinateRange] = None
    surface_nearest: bool = False
    surface_only: bool = False
    surface_suffix: str = "_sfc"
    
    def __post_init__(self):
        """Validate vertical selection parameters."""
        _validate_index_range("index_range", self.index_range)
        _validate_coordinate_range("height_range", self.height_range)
        
        # Additional validation: heights should be non-negative
        if self.height_range is not None and self.height_range[0] < 0:
            raise ValueError("height_range values must be non-negative")
            
        # Warn if both height and index ranges are specified
        if self.height_range is not None and self.index_range is not None:
            import warnings
            warnings.warn("Both height_range and index_range specified. Using index_range (index-based).")
            
        # Business logic validation
        if self.surface_only and not self.surface_nearest:
            raise ValueError("surface_only requires surface_nearest to be True")
    
    @property
    def has_selection(self) -> bool:
        """Check if any vertical selection is defined."""
        return (self.index_range is not None or 
                self.height_range is not None or
                self.surface_nearest)
    
    @property
    def uses_index_selection(self) -> bool:
        """Check if index-based selection is being used."""
        return self.index_range is not None
    
    @property
    def uses_height_selection(self) -> bool:
        """Check if height-based selection is being used."""
        return self.height_range is not None
    
    def get_effective_selection(self) -> Union[IndexRange, CoordinateRange, None]:
        """Get the effective vertical selection (index takes priority)."""
        return self.index_range if self.index_range is not None else self.height_range

# ============================================================================
# Processing Options
# ============================================================================

@dataclass
class ProcessingOptions:
    """
    Data processing options for loading and transformation.
    
    Attributes:
        mask_terrain: Mask values inside terrain
        center_staggered: Center staggered wind variables
        center_suffix: Suffix for centered variables
        chunks: Dask chunking configuration
        engine: xarray backend engine
    """
    mask_terrain: bool = True
    center_staggered: bool = True
    center_suffix: str = "_c"
    chunks: ChunkSetting = "auto"
    engine: Optional[str] = None
    
    def __post_init__(self):
        """Validate processing options."""
        if isinstance(self.chunks, dict):
            for key, value in self.chunks.items():
                if not isinstance(key, str):
                    raise ValueError("Chunk keys must be strings")
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("Chunk values must be positive integers")

# ============================================================================
# Load Parameters Summary
# ============================================================================

@dataclass
class LoadParameters:
    """
    Consolidated loading parameters for the VVM dataset loader.
    
    This class brings together all the various parameter types into a single
    structure for easier passing between functions.
    """
    groups: Optional[Sequence[str]] = None
    variables: Optional[Sequence[str]] = None
    region: Region = field(default_factory=Region)
    time_selection: TimeSelection = field(default_factory=TimeSelection)
    vertical_selection: VerticalSelection = field(default_factory=VerticalSelection)
    processing_options: ProcessingOptions = field(default_factory=ProcessingOptions)
    var_manifest: Optional[Union[str, Path, Mapping[str, Any]]] = None

# ============================================================================
# File and Coordinate Info
# ============================================================================

@dataclass
class FileInfo:
    """Information about a VVM output file."""
    path: Path
    group: str
    index: Optional[int] = None
    timestamp: Optional[np.datetime64] = None

    def __post_init__(self):
        """Validate file information."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.index is not None:
            _validate_index_range("index", (self.index, self.index))  # Single-value validation

@dataclass
class CoordinateInfo:
    """Coordinate system information for the dataset."""
    lon: np.ndarray
    lat: np.ndarray
    x_dim: str
    y_dim: str
    lev: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    z_dim: str = "lev"
    lon_attrs: Dict[str, Any] = field(default_factory=dict)
    lat_attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SliceInfo:
    """Information about spatial and vertical slicing."""
    x_slice: slice
    y_slice: slice
    z_slice: Optional[slice] = None
    periodic_x: bool = False
    periodic_y: bool = False
    
    @property
    def is_full_domain_x(self) -> bool:
        """Check if x slice covers full domain."""
        return self.periodic_x
    
    @property
    def is_full_domain_y(self) -> bool:
        """Check if y slice covers full domain."""
        return self.periodic_y