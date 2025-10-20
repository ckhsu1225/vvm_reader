# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VVM Reader is a Python package for efficiently reading and processing VVM (Vector Vorticity equation cloud-resolving Model) output data into xarray datasets. The package provides a structured, type-safe API for loading atmospheric model output with features like terrain masking, wind centering, and flexible spatial/temporal selection.

## Development Commands

### Environment Setup
The project uses `uv` for dependency management with Python 3.13:
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Testing
Currently, there are no test files in the repository. When adding tests, they should be placed in a `tests/` directory at the project root.

### Installation
```bash
# Development installation
pip install -e .

# Or using uv
uv pip install -e .
```

## Architecture

### Module Organization

The codebase is organized into four main subsystems:

1. **Core** (`src/vvm_reader/core/`): Type definitions and configuration
   - `core_types.py`: Dataclass-based parameter types (`Region`, `TimeSelection`, `VerticalSelection`, `ProcessingOptions`)
   - `config.py`: Constants and configuration (group names, dimension names, file patterns)
   - `exceptions.py`: Custom exception hierarchy

2. **I/O** (`src/vvm_reader/io/`): File operations and dataset loading
   - `file_utils.py`: File discovery, group resolution, filename parsing
   - `dataset_loader.py`: Main orchestration class `VVMDatasetLoader`
   - `manifest.py`: Variable manifest handling for optimized group selection

3. **Coordinates** (`src/vvm_reader/coordinates/`): Coordinate system handling
   - `spatial_handler.py`: Lon/lat/x/y coordinate extraction, regional slicing, coordinate conversion
   - `time_handler.py`: Time coordinate handling, file filtering by time range

4. **Processing** (`src/vvm_reader/processing/`): Data transformations
   - `terrain.py`: Terrain masking and wind centering on staggered grids
   - `vertical.py`: Vertical level selection, surface extraction, fort.98 parsing

### Data Flow

The main data loading pipeline (`VVMDatasetLoader.load_dataset()`) follows these steps:

1. **Parameter Resolution**: Load variable manifest, resolve which groups contain requested variables
2. **Coordinate Setup**: Load TOPO.nc to extract coordinate system and terrain information
3. **Spatial Selection**: Compute slicing information for regional selection (coordinate-based or index-based)
4. **Vertical Selection**: Resolve vertical levels from fort.98 or use index ranges
5. **File Filtering**: Filter NetCDF files by group and optionally by time range
6. **Data Loading**: Load data with spatial/vertical slicing applied
7. **Post-Processing**: Apply terrain masking, center staggered winds, extract surface values

### Key Design Patterns

**Structured Parameters**: The API uses dataclass-based parameter objects rather than keyword arguments. This provides better organization and type safety:
- `Region`: Spatial selection (supports both coordinate and index-based)
- `TimeSelection`: Time range selection (datetime or file indices)
- `VerticalSelection`: Vertical level selection (heights or indices)
- `ProcessingOptions`: Control terrain masking, wind centering, chunking

**Index vs Coordinate Selection**: The API supports both approaches with index-based taking priority when both are specified. Index-based is faster but requires knowing grid indices; coordinate-based is more intuitive but requires coordinate conversion.

**Two-Pass Processing**: Wind centering on staggered grids requires reading extra grid points, then cropping after centering. The loader handles this automatically through `compute_read_slices()` and `crop_dataset_after_centering()`.

**Terrain-Following Coordinates**: VVM uses terrain-following vertical coordinates where `k <= topo` indicates points inside terrain. The package handles this through automatic masking and surface extraction.

## VVM-Specific Details

### File Structure
VVM simulations have this directory layout:
```
simulation_dir/
├── TOPO.nc                    # Terrain topography
├── fort.98                    # Vertical level heights
└── archive/                   # Output files
    ├── sim.L.Dynamic-000001.nc
    ├── sim.L.Thermodynamic-000001.nc
    └── ...
```

### Output Groups
Variables are organized into groups:
- `C.LandSurface`: Land surface variables
- `C.Surface`: Surface variables
- `L.Dynamic`: Wind fields (u, v, w) on staggered grid
- `L.Thermodynamic`: Temperature and moisture (th, qv, qc, qr)
- `L.Radiation`: Radiation variables
- `L.Tracer`: Tracer variables

### Staggered Grid
VVM uses an Arakawa-C staggered grid where u, v, w are on different grid points. Wind centering is handled automatically by `center_staggered_winds()` which averages neighboring points to center variables on the scalar grid.

### Vertical Coordinates
Vertical levels are stored in `fort.98` with heights in meters. The package parses this file to enable height-based vertical selection. Levels use 1-based indexing in VVM but are converted to 0-based for Python.

### Time Coordinates
Time information is stored in each NetCDF file. File indices (e.g., 000001, 000072) can be used for faster selection than datetime-based filtering.

## Working with the Codebase

### Adding New Features

When adding features to the API:
1. Define new parameter types as dataclasses in `core_types.py` if needed
2. Add validation in `__post_init__()` methods
3. Update `LoadParameters` if the feature affects data loading
4. Implement core logic in appropriate module (io/, coordinates/, processing/)
5. Expose through convenience functions in `main.py`
6. Update `__init__.py` exports

### Coordinate Conversions

The package provides bidirectional conversion between coordinates and indices:
- `convert_coordinates_to_indices()`: lon/lat → x/y indices
- `convert_indices_to_coordinates()`: x/y indices → lon/lat
- `convert_time_to_indices()`: datetime range → file index range
- `convert_heights_to_indices()`: height range → vertical indices

These are implemented in `spatial_handler.py`, `time_handler.py`, and `vertical.py` respectively.

### Manifest System

Variable manifests map variables to their containing groups, enabling automatic group selection:
```json
{
  "u": ["L.Dynamic"],
  "th": ["L.Thermodynamic"],
  ...
}
```

Manifests can be auto-generated with `create_variable_manifest()` or manually provided.

### Error Handling

Custom exceptions are defined in `core/exceptions.py`:
- `VVMReaderError`: Base exception
- `SimulationDirectoryError`: Invalid simulation directory
- `GroupNotFoundError`: Requested group not found
- `VariableNotFoundError`: Requested variable not found
- `ManifestError`: Manifest-related errors
- `DataProcessingError`: Processing failures

## Important Constants

From `core/config.py`:
- `KNOWN_GROUPS`: Tuple of recognized VVM output groups
- `WIND_VARIABLES`: ("u", "v", "w") - variables requiring centering
- `VERTICAL_DIM`: "lev"
- `TIME_DIM`: "time"
- `TOPO_VAR`: "topo"
- `FORT98_FILENAME`: "fort.98"

## HPC Environment Notes

This project runs on an HPC system with:
- Intel OneAPI 2022.1 modules (loaded automatically in shell)
- PnetCDF 1.12.2 with HDF5/NetCDF dependencies
- Python 3.13 via `.python-version`

The environment modules are loaded via shell configuration, so commands may show module loading messages which can be ignored.
