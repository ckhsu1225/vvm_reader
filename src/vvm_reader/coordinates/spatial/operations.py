"""
VVM Reader Spatial Dataset Operations

This module provides operations for applying spatial selections and
coordinate assignments to xarray datasets.
"""

from typing import Dict
import xarray as xr

from ...core.config import LON_DIM, LAT_DIM
from ...core.core_types import CoordinateInfo, SliceInfo


# ============================================================================
# Dataset Operations
# ============================================================================

def apply_spatial_selection(
    data: xr.Dataset | xr.DataArray,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo
) -> xr.Dataset | xr.DataArray:
    """
    Apply spatial selection to a dataset or data array.

    Args:
        data: Input dataset or data array (e.g., 2D topo)
        coord_info: Coordinate information
        slice_info: Slice information

    Returns:
        xr.Dataset | xr.DataArray: Spatially selected data
    """
    indexers = {}

    if coord_info.x_dim in data.sizes:
        indexers[coord_info.x_dim] = slice_info.x_slice

    if coord_info.y_dim in data.sizes:
        indexers[coord_info.y_dim] = slice_info.y_slice

    if indexers:
        data = data.isel(indexers)

    return data


def crop_dataset_after_centering(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    crop_offsets: Dict[str, int],
    target_shape: Dict[str, int]
) -> xr.Dataset:
    """
    Crop dataset back to original selection after wind centering.

    Args:
        dataset: Dataset after centering operations
        coord_info: Coordinate information
        crop_offsets: Cropping offsets from centering
        target_shape: Target shape for each dimension

    Returns:
        xr.Dataset: Cropped dataset
    """
    indexers = {}

    if crop_offsets["x"] > 0 and coord_info.x_dim in dataset.sizes:
        nx = target_shape.get(coord_info.x_dim, dataset.sizes[coord_info.x_dim])
        indexers[coord_info.x_dim] = slice(crop_offsets["x"], crop_offsets["x"] + nx)

    if crop_offsets["y"] > 0 and coord_info.y_dim in dataset.sizes:
        ny = target_shape.get(coord_info.y_dim, dataset.sizes[coord_info.y_dim])
        indexers[coord_info.y_dim] = slice(crop_offsets["y"], crop_offsets["y"] + ny)

    if indexers:
        dataset = dataset.isel(indexers)

    return dataset


def assign_spatial_coordinates(
    dataset: xr.Dataset,
    coord_info: CoordinateInfo,
    slice_info: SliceInfo
) -> xr.Dataset:
    """
    Assign lon/lat coordinates to dataset, and preserve xc/yc if present.

    Args:
        dataset: Input dataset
        coord_info: Full coordinate information
        slice_info: Applied slice information

    Returns:
        xr.Dataset: Dataset with assigned coordinates
    """
    # Extract subsets of lon/lat based on slices
    lon_subset = coord_info.lon[slice_info.x_slice]
    lat_subset = coord_info.lat[slice_info.y_slice]

    # Create coordinate arrays with proper dimension names
    lon_coord = xr.DataArray(lon_subset, dims=[coord_info.x_dim])
    if coord_info.lon_attrs:
        lon_coord.attrs.update(coord_info.lon_attrs)

    lat_coord = xr.DataArray(lat_subset, dims=[coord_info.y_dim])
    if coord_info.lat_attrs:
        lat_coord.attrs.update(coord_info.lat_attrs)

    # Prepare coordinate dictionary
    new_coords = {
        LON_DIM: lon_coord,
        LAT_DIM: lat_coord
    }

    # Preserve xc, yc if they exist in the dataset (from VVM nc files)
    # These are Cartesian coordinates in meters, useful for distance calculations
    if 'xc' in dataset.coords:
        new_coords['xc'] = dataset.coords['xc']
    if 'yc' in dataset.coords:
        new_coords['yc'] = dataset.coords['yc']

    return dataset.assign_coords(new_coords)
