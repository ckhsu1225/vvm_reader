"""
VVM Reader Time Coordinate Processing

This module handles time-related coordinate transformations, time range calculations,
and temporal filtering operations.
"""

from typing import List, Optional, Tuple, Sequence
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr

from ..core.config import DATETIME_PRECISION, TIME_DIM
from ..core.core_types import TimeSelection, TimeRange, TimeValue
from ..core.exceptions import ParameterError
from ..io.file_utils import parse_index_from_filename

# ============================================================================
# Time Value Normalization
# ============================================================================

def normalize_time_value(time_value: TimeValue) -> np.datetime64:
    """
    Normalize various time formats to numpy.datetime64.
    
    Args:
        time_value: Time value (str, datetime, or np.datetime64)
        
    Returns:
        np.datetime64: Normalized time value
        
    Raises:
        ParameterError: If time format is invalid
    """
    try:
        if isinstance(time_value, np.datetime64):
            return time_value.astype(f'datetime64[{DATETIME_PRECISION}]')
        
        if isinstance(time_value, datetime):
            return np.datetime64(time_value, DATETIME_PRECISION)
        
        if isinstance(time_value, str):
            dt = datetime.fromisoformat(str(time_value))
            return np.datetime64(dt, DATETIME_PRECISION)
        
        return np.datetime64(time_value, DATETIME_PRECISION)
        
    except Exception as e:
        raise ParameterError("time_value", str(time_value), f"Cannot parse time value: {e}")

def normalize_time_range(time_range: Optional[TimeRange]) -> Tuple[Optional[np.datetime64], Optional[np.datetime64]]:
    """
    Normalize time range to numpy.datetime64 values.
    
    Args:
        time_range: Time range (start, end)
        
    Returns:
        Tuple[Optional[np.datetime64], Optional[np.datetime64]]: Normalized time range
    """
    if time_range is None:
        return None, None
    
    if len(time_range) != 2:
        raise ParameterError("time_range", str(time_range), "Must contain exactly 2 values")
    
    start_time = normalize_time_value(time_range[0]) if time_range[0] is not None else None
    end_time = normalize_time_value(time_range[1]) if time_range[1] is not None else None
    
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ParameterError("time_range", str(time_range), "Start time must be <= end time")
    
    return start_time, end_time

# ============================================================================
# Time Computation
# ============================================================================

def read_time_from_file(file_path: Path) -> np.datetime64:
    """
    Read the primary timestamp from a NetCDF file.

    Args:
        file_path: Path to the NetCDF file

    Returns:
        np.datetime64: Timestamp extracted from the file

    Raises:
        ValueError: If the time coordinate cannot be found or parsed
    """
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            if TIME_DIM in ds.coords:
                time_values = ds.coords[TIME_DIM].values
            elif TIME_DIM in ds:
                time_values = ds[TIME_DIM].values
            else:
                raise ValueError(f"Time coordinate '{TIME_DIM}' not found")

            if np.size(time_values) == 0:
                raise ValueError("Time coordinate is empty")

            primary_time = np.asarray(time_values).flat[0]
            return normalize_time_value(primary_time)

    except Exception as exc:
        raise ValueError(f"Failed to read time from {file_path}: {exc}") from exc

def filter_files_by_time(
    files: Sequence[Path],
    time_selection: TimeSelection
) -> List[Path]:
    """
    Filter files based on time selection criteria.

    Uses index-based filtering when available (faster), falls back to time-based filtering.

    Args:
        files: List of file paths
        time_selection: Time selection parameters

    Returns:
        List[Path]: Filtered files
    """
    result: List[Path] = []

    # Determine filtering method (index-based takes priority)
    if time_selection.uses_index_selection:
        # Fast index-based filtering
        i0, i1 = time_selection.time_index_range

        for file_path in files:
            try:
                index = parse_index_from_filename(file_path)
            except Exception:
                continue

            if not (i0 <= index <= i1):
                continue

            result.append(file_path)

    elif time_selection.uses_time_selection:
        # Time-based filtering (requires time computation for each file)
        tr_start, tr_end = normalize_time_range(time_selection.time_range)

        for file_path in files:
            timestamp = read_time_from_file(file_path)

            if tr_start is not None and timestamp < tr_start:
                continue
            if tr_end is not None and timestamp > tr_end:
                continue

            result.append(file_path)
    else:
        # No time filtering - include all files
        for file_path in files:
            result.append(file_path)

    # Sort by timestamp
    return sorted(result)
