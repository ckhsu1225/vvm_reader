# VVM Reader

A Python package for efficiently reading and processing VVM (Vector Vorticity equation cloud-resolving Model) output data into xarray datasets.

## Features

- **Structured Parameter Interface**: Clean, type-safe API with organized parameter classes
- **Flexible Spatial Selection**: Support both coordinate-based (lon/lat) and index-based (x/y) selection
- **Smart Time Selection**: Choose by datetime or file indices for optimal performance  
- **Intelligent Vertical Selection**: Select by height or level indices with automatic fort.98 integration
- **Automatic Terrain Processing**: Built-in terrain masking and surface extraction
- **Wind Field Centering**: Automatic centering of staggered wind variables
- **Variable Manifests**: Optimized loading with automatic group selection
- **Dask Integration**: Memory-efficient chunked loading for large datasets

## Installation

```bash
pip install vvm-reader
```

## Quick Start

### Basic Usage

```python
import vvm_reader as vvm

# Load all data from a simulation
ds = vvm.open_vvm_dataset("/path/to/simulation")

# Load specific variables
ds = vvm.open_vvm_dataset(
    "/path/to/simulation", 
    variables=["u", "v", "w", "th"]
)
```

### Regional Selection

```python
# Coordinate-based selection
region = vvm.Region(
    lon_range=(120.0, 122.0),
    lat_range=(23.0, 25.0)
)

ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    variables=["u", "v", "w"],
    region=region
)

# Index-based selection (more precise)
region = vvm.Region(
    x_range=(100, 200),
    y_range=(50, 150)
)

ds = vvm.open_vvm_dataset("/path/to/simulation", region=region)
```

### Time Selection

```python
from datetime import datetime

# Time-based selection
time_sel = vvm.TimeSelection(
    time_range=(
        datetime(2001, 5, 20, 12, 0),
        datetime(2001, 5, 20, 18, 0)
    )
)

# Index-based selection (faster)
time_sel = vvm.TimeSelection(
    time_index_range=(72, 108)  # Files 000072.nc to 000108.nc
)

ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    time_selection=time_sel
)
```

### Vertical Selection

```python
# Height-based selection
vert_sel = vvm.VerticalSelection(
    height_range=(0, 5000)  # 0 to 5km height
)

# Index-based selection (faster)
vert_sel = vvm.VerticalSelection(
    index_range=(1, 25)  # Levels 1 to 25
)

ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    vertical_selection=vert_sel
)
```

## Convenience Functions

### Surface Data

```python
# Automatically detect vertical range needed for surface extraction
ds = vvm.load_surface_data(
    "/path/to/simulation", 
    variables=["u", "v", "th"]
)
```

### Quick Loading

```python
# Load first 10 time steps
ds = vvm.quick_load(
    "/path/to/simulation",
    variables=["u", "v", "w"], 
    time_steps=10
)
```

### Regional Analysis

```python
# Load specific region by indices
ds = vvm.load_indices(
    "/path/to/simulation",
    x_range=(100, 200),
    y_range=(50, 150),
    variables=["th", "qv"]
)
```

### Time Series Analysis

```python
# Load time series with spatial averaging
ds = vvm.load_time_series(
    "/path/to/simulation",
    variables=["th", "qv"],
    spatial_mean=True
)
```

## Advanced Features

### Coordinate Conversion

```python
# Convert coordinates to indices for precise selection
indices = vvm.convert_coordinates_to_indices(
    "/path/to/simulation",
    lon_range=(120.5, 121.5),
    lat_range=(23.5, 24.5)
)

print(f"X indices: {indices['x_range']}")  # (120, 180)
print(f"Y indices: {indices['y_range']}")  # (80, 120)

# Use precise indices for loading
ds = vvm.load_indices(
    "/path/to/simulation",
    indices['x_range'], 
    indices['y_range']
)
```

### Time Conversion

```python
from datetime import datetime

# Convert time range to file indices
time_indices = vvm.convert_time_to_indices(
    "/path/to/simulation",
    time_range=(
        datetime(2001, 5, 20, 12),
        datetime(2001, 5, 20, 18)
    )
)

print(f"File indices: {time_indices['time_index_range']}")  # (72, 108)
```

### Height Conversion

```python
# Convert height range to vertical indices  
height_indices = vvm.convert_heights_to_indices(
    "/path/to/simulation",
    height_range=(0, 5000)  # 0-5km
)

print(f"Vertical indices: {height_indices['index_range']}")  # (1, 25)
```

### Simulation Information

```python
# Get simulation overview
info = vvm.get_simulation_info("/path/to/simulation")
print(f"Available groups: {info['available_groups']}")
print(f"Total files: {info['total_files']}")

# Get coordinate system info
coord_info = vvm.get_coordinate_info("/path/to/simulation")
print(f"Grid size: {coord_info['nx']} x {coord_info['ny']}")
print(f"Resolution: {coord_info['dx_mean']:.3f}°")

# Get terrain information
terrain_info = vvm.get_terrain_info("/path/to/simulation")
print(f"Terrain range: {terrain_info['min_level']} to {terrain_info['max_level']}")
print(f"Ocean fraction: {terrain_info['ocean_fraction']:.1%}")
```

## Processing Options

### Custom Processing

```python
# Disable terrain masking and wind centering
proc_opts = vvm.ProcessingOptions(
    mask_terrain=False,
    center_staggered=False,
    chunks={"time": 1, "y": 100, "x": 100}
)

ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    processing_options=proc_opts
)
```

### Surface Extraction

```python
# Extract surface values only (removes vertical dimension)
vert_sel = vvm.VerticalSelection(
    surface_nearest=True,
    surface_only=True
)

# Or add surface variables alongside 3D variables
vert_sel = vvm.VerticalSelection(
    surface_nearest=True,
    surface_only=False,
    surface_suffix="_sfc"  # Creates u_sfc, v_sfc, etc.
)
```

## Understanding VVM Data Structure

### Output Groups

VVM organizes variables into groups:
- `C.LandSurface`: Land surface variables
- `C.Surface`: Surface variables  
- `L.Dynamic`: Wind and dynamic variables (u, v, w)
- `L.Thermodynamic`: Temperature and moisture (th, qv, qc, qr)
- `L.Radiation`: Radiation variables
- `L.Tracer`: Tracer variables

### Coordinate System

- **Horizontal**: Staggered Arakawa-C grid
- **Vertical**: Terrain-following coordinates with 1-based indexing
- **Time**: File-based with configurable intervals (default 10 minutes)
- **Terrain**: Integer levels where k ≤ topo indicates inside terrain

### File Naming Convention

```
simulation_name.GROUP_NAME-XXXXXX.nc
```

Examples:
- `sim.L.Dynamic-000001.nc`: First output file for dynamics
- `sim.L.Thermodynamic-000036.nc`: 36th output file for thermodynamics

## Performance Tips

1. **Use index-based selection** when possible for better performance
2. **Load surface data efficiently** with `load_surface_data()` 
3. **Convert coordinates to indices** for repeated analysis of the same region
4. **Use appropriate chunking** for your memory constraints
5. **Filter variables early** by specifying only needed variables
6. **Leverage manifests** for optimal group selection

## API Reference

### Main Functions

- `open_vvm_dataset()`: Main loading function with structured parameters
- `quick_load()`: Fast loading for common cases
- `load_surface_data()`: Surface data with automatic vertical range detection
- `load_region()`: Regional data loading
- `load_indices()`: Index-based regional loading  
- `load_time_series()`: Time series analysis

### Parameter Classes

- `Region`: Spatial selection (coordinates or indices)
- `TimeSelection`: Time selection (datetime or file indices) 
- `VerticalSelection`: Vertical selection (heights or levels)
- `ProcessingOptions`: Data processing controls

### Utility Functions

- `get_simulation_info()`: Simulation overview
- `get_coordinate_info()`: Grid and coordinate information
- `get_terrain_info()`: Terrain statistics
- `convert_*_to_*()`: Various coordinate conversion functions
- `list_available_simulations()`: Find simulations in a directory

## Examples

### Multi-Regional Analysis

```python
# Analyze multiple regions efficiently
regions = {
    'taipei': vvm.Region(lon_range=(121.3, 121.8), lat_range=(24.8, 25.3)),
    'taichung': vvm.Region(lon_range=(120.4, 120.9), lat_range=(24.0, 24.5)),
    'kaohsiung': vvm.Region(lon_range=(120.1, 120.6), lat_range=(22.4, 22.9))
}

data = {}
for name, region in regions.items():
    data[name] = vvm.load_surface_data(
        "/path/to/simulation",
        variables=["th", "qv"],
        region=region
    )
```

### Time Series Analysis

```python
# Domain-averaged time series
ts_data = vvm.load_time_series(
    "/path/to/simulation",
    variables=["th", "qv", "w"],
    spatial_mean=True
)

# Plot domain-averaged vertical velocity
import matplotlib.pyplot as plt
ts_data.w.mean(dim='lev').plot()
plt.title('Domain-averaged Vertical Velocity')
plt.show()
```

### Custom Processing Pipeline

```python
# Load raw data without processing
raw_ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    variables=["u", "v", "th"],
    processing_options=vvm.ProcessingOptions(
        mask_terrain=False,
        center_staggered=False
    )
)

# Apply custom processing
# ... your custom processing logic here ...
```

## Contributing

Contributions are welcome! Please see our contributing guidelines and submit pull requests or issues on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use VVM Reader in your research, please cite:

```bibtex
@software{vvm_reader,
  title={VVM Reader: A Python Package for VVM Model Output Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/vvm-reader}
}
```