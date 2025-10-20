"""
VVM Reader Main Dataset Loader

This module contains the core dataset loading functionality that orchestrates
all the processing steps to load VVM data into xarray datasets.
"""

import logging
import xarray as xr
from pathlib import Path

from ..core.config import STAGGERED_VARIABLES, TIME_DIM
from ..core.core_types import LoadParameters
from ..core.exceptions import (
    NoDataError, validate_simulation_directory
)

# Get logger for this module
logger = logging.getLogger('vvm_reader.io.dataset_loader')

from ..io.file_utils import (
    resolve_groups_to_load, 
    get_variables_for_group, filter_files_by_groups
)
from ..io.manifest import load_or_create_manifest

from ..coordinates.time_handler import filter_files_by_time

from ..coordinates.spatial import (
    load_topo_dataset, extract_coordinates_from_topo, compute_surface_topo_levels,
    extract_terrain_mask, compute_regional_slices, compute_centering_slices, apply_spatial_selection,
    crop_dataset_after_centering, assign_spatial_coordinates
)

from ..processing.vertical import (
    resolve_vertical_slice, extend_vertical_slice_for_centering,
    extract_surface_nearest_values, apply_vertical_selection,
    crop_vertical_after_centering, ensure_vertical_coordinate_in_meters
)
from ..processing.terrain import (
    apply_terrain_mask, center_staggered_variables
)

# ============================================================================
# Main Dataset Loader Class
# ============================================================================

class VVMDatasetLoader:
    """
    Main VVM dataset loader class.
    
    This class orchestrates the entire data loading process including:
    - Parameter validation and resolution
    - File discovery and filtering
    - Coordinate system setup
    - Data loading and processing
    - Post-processing operations
    """
    
    def __init__(self, sim_dir: Path):
        """
        Initialize the dataset loader.
        
        Args:
            sim_dir: Simulation directory path
        """
        self.sim_dir = Path(sim_dir)
        self.archive_dir = validate_simulation_directory(self.sim_dir)
        
    def load_dataset(self, params: LoadParameters) -> xr.Dataset:
        """
        Load VVM dataset with specified parameters.
        
        Args:
            params: Consolidated loading parameters
            
        Returns:
            xr.Dataset: Loaded and processed dataset
        """
        # Step 1: Load manifest and resolve groups/variables
        manifest = load_or_create_manifest(params.var_manifest, self.sim_dir)
        groups_to_load = resolve_groups_to_load(
            self.sim_dir, params.groups, params.variables, manifest
        )
        
        if not groups_to_load:
            raise NoDataError("No valid groups found to load")
        
        # Step 2: Setup coordinate system
        coord_info, sfc_level, terrain_mask = self._setup_coordinates(params.processing_options.engine)
        slice_info = compute_regional_slices(coord_info, params.region)
        
        # Step 3: Resolve vertical selection
        vertical_slice = resolve_vertical_slice(self.sim_dir, params.vertical_selection)
        
        # Step 4: Determine if centering is needed
        needs_centering = self._check_centering_needed(
            params.variables, params.processing_options.center_staggered
        )
        
        # Step 5: Compute extended slices for centering if needed
        read_slice_info, crop_offsets = self._compute_read_slices(
            slice_info, needs_centering, params.variables
        )

        # Handle vertical extension for z-staggered variables (w, eta, xi)
        needs_z_centering = any(
            needs_centering.get(var, False) and (params.variables is None or var in params.variables)
            for var in ["w", "eta", "xi"]
        )
        vertical_read_slice, vertical_crop_offset, vertical_target_length = (
            extend_vertical_slice_for_centering(vertical_slice, needs_z_centering)
        )
        
        # Step 6: Load data from all groups
        group_datasets = self._load_group_datasets(
            groups_to_load, params, manifest, coord_info,
            read_slice_info, vertical_read_slice
        )
        
        if not group_datasets:
            raise NoDataError("No data loaded from any group")
        
        # Step 7: Merge datasets from all groups
        dataset = xr.merge(group_datasets, compat="no_conflicts")
        
        # Step 8: Post-processing pipeline
        dataset = self._post_process_dataset(
            dataset, params, coord_info, slice_info, sfc_level,
            crop_offsets, vertical_crop_offset, vertical_target_length,
            needs_centering, read_slice_info, terrain_mask,
            vertical_slice, vertical_read_slice
        )
        
        return dataset
    
    def _setup_coordinates(self, engine):
        """Setup coordinate system from TOPO.nc."""
        topo_ds = load_topo_dataset(self.sim_dir, engine)
        coord_info = extract_coordinates_from_topo(topo_ds)
        terrain_mask = extract_terrain_mask(topo_ds)
        sfc_level = compute_surface_topo_levels(terrain_mask['mask'])
        topo_ds.close()
        return coord_info, sfc_level, terrain_mask
    
    def _check_centering_needed(self, variables, center_staggered):
        """Check which staggered variables need centering (winds and vorticities)."""
        if not center_staggered:
            return {}

        needs_centering = {}
        if variables is None:
            # If no specific variables requested, assume all staggered vars might be present
            needs_centering = {var: True for var in STAGGERED_VARIABLES}
        else:
            for var in STAGGERED_VARIABLES:
                needs_centering[var] = var in variables

        return needs_centering
    
    def _compute_read_slices(self, slice_info, needs_centering, variables):
        """Compute extended slices for reading with centering halos."""
        # Check if any variable needs x-direction halo
        # u, zeta, eta need x-halo
        needs_x_halo = any(
            needs_centering.get(var, False) and (variables is None or var in variables)
            for var in ["u", "zeta", "eta"]
        )

        # Check if any variable needs y-direction halo
        # v, zeta, xi need y-halo
        needs_y_halo = any(
            needs_centering.get(var, False) and (variables is None or var in variables)
            for var in ["v", "zeta", "xi"]
        )

        return compute_centering_slices(
            slice_info, needs_x_halo, needs_y_halo
        )
    
    def _load_group_datasets(
        self, groups_to_load, params, manifest, coord_info,
        read_slice_info, vertical_read_slice
    ):
        """Load datasets from all specified groups."""
        group_datasets = []
        
        for group in groups_to_load:
            try:
                group_ds = self._load_single_group(
                    group, params, manifest, coord_info,
                    read_slice_info, vertical_read_slice
                )
                if group_ds is not None:
                    group_datasets.append(group_ds)
            except Exception as e:
                logger.warning("Failed to load group %s: %s", group, e)
                continue
        
        return group_datasets
    
    def _load_single_group(
        self, group, params, manifest, coord_info,
        read_slice_info, vertical_read_slice
    ):
        """Load dataset from a single output group."""
        # Get files for this group
        group_files_dict = filter_files_by_groups(self.archive_dir, [group])
        files = group_files_dict.get(group, [])
        
        if not files:
            return None
        
        # Filter files by time
        files = filter_files_by_time(files, params.time_selection)

        if not files:
            return None

        # Determine variables to keep for this group
        keep_vars = None
        if params.variables is not None and manifest is not None:
            keep_vars = get_variables_for_group(group, params.variables, manifest)
            if keep_vars is not None and not keep_vars:
                return None

        # Load and concatenate files
        file_datasets = []
        for file_path in files:
            file_ds = self._load_single_file(
                file_path, keep_vars, coord_info, read_slice_info,
                vertical_read_slice, params.processing_options
            )
            if file_ds is not None:
                file_datasets.append(file_ds)

        if not file_datasets:
            return None

        return xr.concat(file_datasets, dim=TIME_DIM)
    
    def _load_single_file(
        self, file_path, keep_vars, coord_info, slice_info,
        vertical_slice, processing_options
    ):
        """Load and process a single file."""
        ds = None
        try:
            # Open dataset with chunking
            ds = xr.open_dataset(
                file_path,
                chunks=processing_options.chunks,
                engine=processing_options.engine
            )

            # Filter variables if specified
            if keep_vars is not None:
                present_vars = [v for v in keep_vars if v in ds.data_vars]
                if not present_vars:
                    ds.close()
                    return None
                ds = ds[present_vars]

            # Apply spatial selection
            ds = apply_spatial_selection(ds, coord_info, slice_info)

            # Apply vertical selection
            ds = apply_vertical_selection(ds, vertical_slice)
            return ds

        except Exception:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
            return None
    
    def _post_process_dataset(
        self, dataset, params, coord_info, slice_info, sfc_level,
        crop_offsets, vertical_crop_offset, vertical_target_length,
        needs_centering, read_slice_info, terrain_mask,
        vertical_slice, vertical_read_slice
    ):
        """Apply all post-processing operations."""
        
        mask_slice = read_slice_info if read_slice_info is not None else slice_info

        terrain_mask_center = apply_spatial_selection(terrain_mask, coord_info, mask_slice)
        terrain_mask_center = apply_vertical_selection(
            terrain_mask_center, vertical_read_slice
        )["mask"]

        terrain_mask_final = apply_spatial_selection(terrain_mask, coord_info, slice_info)
        terrain_mask_final = apply_vertical_selection(
            terrain_mask_final, vertical_slice
        )["mask"]

        # Step 1: Center staggered variables (winds and vorticities) if requested
        created_center_vars = []
        if params.processing_options.center_staggered and any(needs_centering.values()):
            dataset, created_center_vars = center_staggered_variables(
                dataset, coord_info, mask_slice,
                terrain_mask_center,
                params.processing_options.center_suffix,
            )

        # Step 2: Crop back to original selection if halos were added
        if any(crop_offsets.values()):
            target_shape = {
                coord_info.x_dim: slice_info.x_slice.stop - (slice_info.x_slice.start or 0),
                coord_info.y_dim: slice_info.y_slice.stop - (slice_info.y_slice.start or 0)
            }
            dataset = crop_dataset_after_centering(
                dataset, coord_info, crop_offsets, target_shape
            )

        if vertical_crop_offset > 0:
            dataset = crop_vertical_after_centering(
                dataset, vertical_crop_offset, vertical_target_length
            )

        # Step 3: Apply terrain masking
        if params.processing_options.mask_terrain:
            dataset = apply_terrain_mask(
                dataset, terrain_mask_final
            )

        # Step 4: Assign spatial coordinates
        dataset = assign_spatial_coordinates(dataset, coord_info, slice_info)
        dataset = ensure_vertical_coordinate_in_meters(dataset)

        # Step 5: Extract surface values if requested
        if params.vertical_selection.surface_nearest:
            sfc_level = apply_spatial_selection(
                sfc_level, coord_info, slice_info
            )["surface_level"]
            dataset = extract_surface_nearest_values(
                dataset, sfc_level, params.vertical_selection
            )

        # Step 6: Filter to requested variables only
        if params.variables is not None:
            requested_vars = list(params.variables)
            # Always keep created centered variables
            for var in created_center_vars:
                if var not in requested_vars:
                    requested_vars.append(var)
            
            # Keep only variables that exist in the dataset
            keep_vars = [v for v in requested_vars if v in dataset.data_vars]
            if keep_vars:
                dataset = dataset[keep_vars]
        
        # Step 7: Ensure time is sorted
        if TIME_DIM in dataset.coords:
            dataset = dataset.sortby(TIME_DIM)
        
        return dataset

# ============================================================================
# Convenience Functions
# ============================================================================

def load_vvm_dataset(sim_dir: Path, params: LoadParameters) -> xr.Dataset:
    """
    Convenience function to load a VVM dataset.
    
    Args:
        sim_dir: Simulation directory path
        params: Loading parameters
        
    Returns:
        xr.Dataset: Loaded dataset
    """
    loader = VVMDatasetLoader(sim_dir)
    return loader.load_dataset(params)