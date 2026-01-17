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
TimeIndices = Sequence[int]
TimeValues = Sequence[TimeValue]
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

    Supports continuous range selection or arbitrary time/index selection.
    Priority order: time_indices > time_index_range > time_values > time_range

    Attributes:
        time_range: Time range (start_time, end_time) - continuous interval
        time_index_range: File index range (start_index, end_index) - continuous interval
        time_indices: Arbitrary list of file indices (e.g., [0, 5, 10, 20])
        time_values: Arbitrary list of time values (e.g., specific datetimes)

    Examples:
        # Continuous time range
        TimeSelection(time_range=(datetime(2001, 5, 20, 12, 0), datetime(2001, 5, 20, 18, 0)))

        # Continuous index range
        TimeSelection(time_index_range=(72, 108))

        # Arbitrary indices (e.g., every 10th file)
        TimeSelection(time_indices=[0, 10, 20, 30, 40])

        # Arbitrary time values
        TimeSelection(time_values=[datetime(2001, 5, 20, 12, 0), datetime(2001, 5, 20, 18, 0)])
    """
    time_range: Optional[TimeRange] = None
    time_index_range: Optional[IndexRange] = None
    time_indices: Optional[TimeIndices] = None
    time_values: Optional[TimeValues] = None

    def __post_init__(self):
        """Validate time selection parameters."""
        _validate_index_range("time_index_range", self.time_index_range)

        # Validate time_indices
        if self.time_indices is not None:
            if not isinstance(self.time_indices, (list, tuple, np.ndarray, Sequence)):
                raise TypeError("time_indices must be a sequence (list, tuple, or array)")
            if len(self.time_indices) == 0:
                raise ValueError("time_indices must not be empty")
            for idx in self.time_indices:
                if not isinstance(idx, (int, np.integer)):
                    raise TypeError(f"time_indices must contain only integers, got {type(idx)}")
                if idx < 0:
                    raise ValueError(f"time_indices must contain only non-negative integers, got {idx}")

        # Validate time_values
        if self.time_values is not None:
            if not isinstance(self.time_values, (list, tuple, np.ndarray, Sequence)):
                raise TypeError("time_values must be a sequence (list, tuple, or array)")
            if len(self.time_values) == 0:
                raise ValueError("time_values must not be empty")

        # Warn if multiple selection methods are specified
        specified = sum([
            self.time_range is not None,
            self.time_index_range is not None,
            self.time_indices is not None,
            self.time_values is not None
        ])

        if specified > 1:
            import warnings
            priority_msg = "Multiple time selections specified. Priority: time_indices > time_index_range > time_values > time_range"
            warnings.warn(priority_msg)

    @property
    def has_selection(self) -> bool:
        """Check if any time selection is defined."""
        return (self.time_range is not None or
                self.time_index_range is not None or
                self.time_indices is not None or
                self.time_values is not None)

    @property
    def uses_index_selection(self) -> bool:
        """Check if continuous index-based selection is being used."""
        return self.time_index_range is not None and self.time_indices is None

    @property
    def uses_time_selection(self) -> bool:
        """Check if continuous time-based selection is being used."""
        return (self.time_range is not None and
                self.time_indices is None and
                self.time_values is None)

    @property
    def uses_arbitrary_indices(self) -> bool:
        """Check if arbitrary index selection is being used."""
        return self.time_indices is not None

    @property
    def uses_arbitrary_times(self) -> bool:
        """Check if arbitrary time value selection is being used."""
        return self.time_values is not None and self.time_indices is None

    def get_effective_selection(self) -> Union[TimeIndices, IndexRange, TimeValues, TimeRange, None]:
        """
        Get the effective time selection based on priority.
        Priority: time_indices > time_index_range > time_values > time_range
        """
        if self.time_indices is not None:
            return self.time_indices
        if self.time_index_range is not None:
            return self.time_index_range
        if self.time_values is not None:
            return self.time_values
        return self.time_range

# ============================================================================
# Vertical Selection
# ============================================================================

@dataclass
class VerticalSelection:
    """
    Vertical level selection and processing parameters.
    
    Supports continuous range selection or arbitrary level selection.
    Priority order: level_indices > heights > index_range > height_range
    
    Attributes:
        index_range: Vertical index range (start_level, end_level) - continuous
        height_range: Height range (min_height, max_height) in meters - continuous
        level_indices: Arbitrary list of level indices (e.g., [0, 5, 10, 20])
        heights: Arbitrary list of heights in meters (e.g., [500, 1000, 3000])
        surface_nearest: Extract surface-nearest values
        surface_only: Keep only surface values (remove vertical dimension)
        surface_suffix: Suffix for surface variables
    """
    index_range: Optional[IndexRange] = None
    height_range: Optional[CoordinateRange] = None
    level_indices: Optional[Sequence[int]] = None
    heights: Optional[Sequence[float]] = None
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
        
        # Validate level_indices
        if self.level_indices is not None:
            if not isinstance(self.level_indices, (list, tuple, np.ndarray, Sequence)):
                raise TypeError("level_indices must be a sequence (list, tuple, or array)")
            if len(self.level_indices) == 0:
                raise ValueError("level_indices must not be empty")
            for idx in self.level_indices:
                if not isinstance(idx, (int, np.integer)):
                    raise TypeError(f"level_indices must contain only integers, got {type(idx)}")
                if idx < 0:
                    raise ValueError(f"level_indices must contain only non-negative integers, got {idx}")
        
        # Validate heights
        if self.heights is not None:
            if not isinstance(self.heights, (list, tuple, np.ndarray, Sequence)):
                raise TypeError("heights must be a sequence (list, tuple, or array)")
            if len(self.heights) == 0:
                raise ValueError("heights must not be empty")
            for h in self.heights:
                if not isinstance(h, (int, float, np.integer, np.floating)):
                    raise TypeError(f"heights must contain only numbers, got {type(h)}")
                if h < 0:
                    raise ValueError(f"heights must contain only non-negative values, got {h}")
            
        # Warn if multiple selection methods are specified
        specified = sum([
            self.index_range is not None,
            self.height_range is not None,
            self.level_indices is not None,
            self.heights is not None
        ])
        
        if specified > 1:
            import warnings
            warnings.warn(
                "Multiple vertical selections specified. "
                "Priority: level_indices > heights > index_range > height_range"
            )
            
        # Business logic validation
        if self.surface_only and not self.surface_nearest:
            raise ValueError("surface_only requires surface_nearest to be True")
    
    @property
    def has_selection(self) -> bool:
        """Check if any vertical selection is defined."""
        return (self.index_range is not None or 
                self.height_range is not None or
                self.level_indices is not None or
                self.heights is not None or
                self.surface_nearest)
    
    @property
    def uses_index_selection(self) -> bool:
        """Check if continuous index-based selection is being used."""
        return self.index_range is not None and self.level_indices is None
    
    @property
    def uses_height_selection(self) -> bool:
        """Check if continuous height-based selection is being used."""
        return (self.height_range is not None and 
                self.level_indices is None and 
                self.heights is None)
    
    @property
    def uses_arbitrary_indices(self) -> bool:
        """Check if arbitrary index selection is being used."""
        return self.level_indices is not None
    
    @property
    def uses_arbitrary_heights(self) -> bool:
        """Check if arbitrary height selection is being used."""
        return self.heights is not None and self.level_indices is None
    
    @property
    def uses_arbitrary_selection(self) -> bool:
        """Check if any arbitrary (non-contiguous) selection is being used."""
        return self.level_indices is not None or self.heights is not None
    
    def get_effective_selection(self) -> Union[Sequence[int], Sequence[float], IndexRange, CoordinateRange, None]:
        """
        Get the effective vertical selection based on priority.
        Priority: level_indices > heights > index_range > height_range
        """
        if self.level_indices is not None:
            return self.level_indices
        if self.heights is not None:
            return self.heights
        if self.index_range is not None:
            return self.index_range
        return self.height_range


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
        add_terrain_height: Add terrain height (in meters) to dataset
        add_reference_profiles: Add reference state profiles to dataset
        chunks: Dask chunking configuration
        engine: xarray backend engine
    """
    mask_terrain: bool = True
    center_staggered: bool = True
    center_suffix: str = "_c"
    add_terrain_height: bool = False
    add_reference_profiles: bool = False
    add_coriolis_parameter: bool = True
    chunks: ChunkSetting = "auto"
    engine: Optional[str] = None

    def __post_init__(self):
        """Validate processing options."""
        if isinstance(self.chunks, dict):
            for key, value in self.chunks.items():
                if not isinstance(key, str):
                    raise ValueError("Chunk keys must be strings")
                if not isinstance(value, int) or value == 0 or value < -1:
                    raise ValueError("Chunk values must be positive integers or -1 (no chunking)")

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