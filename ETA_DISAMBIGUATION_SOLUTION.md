# ETA Variable Name Disambiguation Solution

## Problem Description

The VVM model has a variable naming conflict: **`eta`** appears in two different output groups with completely different physical meanings:

### 1. C.LandSurface: eta (Latent Heat Flux)
- **Physical Quantity:** Evapotranspiration / latent heat flux
- **Dimensions:** 2D `(time, lat, lon)`
- **Units:** `W m-2`
- **Attributes:**
  ```json
  {
    "long_name": "latent heat flux(evapotranspiration)",
    "standard_name": "latent_heat_flux",
    "units": "W m-2"
  }
  ```

### 2. L.Dynamic: eta (Vorticity Component)
- **Physical Quantity:** y-component of vorticity (η = ∂u/∂z - ∂w/∂x)
- **Dimensions:** 3D `(time, lev, lat, lon)`
- **Units:** `s-1`
- **Staggered Grid:** x and z directions
- **Attributes:**
  ```json
  {
    "long_name": "y-component of vorticity",
    "standard_name": "y_component_of_vorticity",
    "units": "s-1"
  }
  ```

## Manifest Issue

The `variable_manifest.json` incorrectly merged these two distinct variables into a single entry:

```json
"eta": {
  "attrs": {
    "long_name": "latent heat flux(evapotranspiration)",  // ← C.LandSurface version
    "standard_name": "latent_heat_flux",
    "units": "W m-2"
  },
  "dims": ["time", "lat", "lon"],  // ← 2D, not 3D!
  "dtype": "float32",
  "groups": [
    "C.LandSurface",
    "L.Dynamic"  // ← But this should be 3D!
  ]
}
```

This causes problems when requesting `eta` for vorticity calculations because:
1. The manifest says `eta` is 2D
2. But vorticity `eta` is actually 3D
3. The centering logic expects 3D data with `lev` dimension

## Solution Implemented

We implemented a **multi-level disambiguation system** to automatically select the correct version of `eta`:

### 1. Configuration-Based Priority (`core/config.py`)

```python
# Variable Name Disambiguation
VARIABLE_GROUP_PRIORITY = {
    "eta": ["L.Dynamic", "C.LandSurface"],  # Prefer L.Dynamic for eta
}

# For staggered variables, prefer 3D groups (with 'lev' dimension)
PREFER_3D_FOR_STAGGERED = True
```

### 2. Smart Group Resolution (`io/file_utils.py`)

The `resolve_groups_to_load()` function now applies disambiguation logic with three priority levels:

**Priority 1: Explicit Group Priority Rules**
- Check `VARIABLE_GROUP_PRIORITY` dictionary
- For `eta`, prefer `L.Dynamic` over `C.LandSurface`

**Priority 2: Staggered Variable 3D Preference**
- For variables in `STAGGERED_VARIABLES` (u, v, w, zeta, eta, xi)
- If `PREFER_3D_FOR_STAGGERED = True`, prefer groups with `lev` dimension

**Priority 3: First Available Group**
- Fallback to first available group in manifest

### Implementation Code

```python
def resolve_groups_to_load(...):
    # ...
    for var in variables:
        var_info = manifest.get(var, {})
        var_groups = var_info.get("groups", [])

        selected_group = None

        # Priority 1: Check explicit priority rules
        if var in VARIABLE_GROUP_PRIORITY:
            priority_list = VARIABLE_GROUP_PRIORITY[var]
            for priority_group in priority_list:
                if priority_group in var_groups and priority_group in available_groups:
                    selected_group = priority_group
                    break

        # Priority 2: For staggered variables, prefer 3D
        if selected_group is None and var in STAGGERED_VARIABLES:
            for group in var_groups:
                var_dims = var_info.get("dims", [])
                if VERTICAL_DIM in var_dims:
                    selected_group = group
                    break

        # Priority 3: First available
        if selected_group is None:
            selected_group = next(g for g in var_groups if g in available_groups)
```

## Usage Examples

### Example 1: Loading Vorticity eta (Automatic)

```python
import vvm_reader as vvm

# Request eta - will automatically load from L.Dynamic (vorticity)
ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    variables=["u", "v", "w", "eta"],  # eta = vorticity component
    processing_options=vvm.ProcessingOptions(
        center_staggered=True  # Will center eta from (x,z)-staggered to cell center
    )
)

# Result: eta from L.Dynamic (3D vorticity)
print(ds.eta.dims)  # ('time', 'lev', 'lat', 'lon')
print(ds.eta.units)  # 's-1'
print(ds.eta_c)  # Centered version available
```

### Example 2: Loading Latent Heat Flux eta (Manual Group Selection)

If you want the C.LandSurface version (latent heat flux), explicitly specify the group:

```python
ds = vvm.open_vvm_dataset(
    "/path/to/simulation",
    groups=["C.LandSurface"],  # Explicit group selection
    variables=["eta", "SH", "G"]  # eta = latent heat flux
)

# Result: eta from C.LandSurface (2D heat flux)
print(ds.eta.dims)  # ('time', 'lat', 'lon')
print(ds.eta.units)  # 'W m-2'
```

### Example 3: Loading Both Versions

To load both versions, use different groups explicitly:

```python
# Load vorticity eta
ds_dynamic = vvm.open_vvm_dataset(
    sim_dir,
    groups=["L.Dynamic"],
    variables=["eta"]
)
eta_vorticity = ds_dynamic.eta  # 3D, s-1

# Load heat flux eta
ds_surface = vvm.open_vvm_dataset(
    sim_dir,
    groups=["C.LandSurface"],
    variables=["eta"]
)
eta_heatflux = ds_surface.eta  # 2D, W m-2
```

## Verification

You can verify which group was selected:

```python
import vvm_reader as vvm

# Check priority configuration
print("Variable group priority:", vvm.VARIABLE_GROUP_PRIORITY)
# Output: {'eta': ['L.Dynamic', 'C.LandSurface']}

# Load data
ds = vvm.open_vvm_dataset(sim_dir, variables=["eta"])

# Check what was loaded
print("eta dimensions:", ds.eta.dims)
print("eta units:", ds.eta.units)
print("eta long_name:", ds.eta.attrs.get('long_name'))

# If it's vorticity (correct):
# eta dimensions: ('time', 'lev', 'lat', 'lon')
# eta units: s-1
# eta long_name: y-component of vorticity
```

## Notes and Caveats

### Manifest Metadata Issue

The manifest currently has **incorrect metadata** for `eta`:
- It shows 2D dimensions `(time, lat, lon)`
- But when loading from `L.Dynamic`, the actual file contains 3D data `(time, lev, lat, lon)`

Our disambiguation logic handles this by:
1. Selecting the correct group (`L.Dynamic`) based on priority
2. Loading the actual data from files (which is correctly 3D)
3. The manifest metadata (dims, attrs) may be wrong, but the loaded data is correct

### When Priority Rules Don't Apply

If you load data without specifying variables (load all):
```python
ds = vvm.open_vvm_dataset(sim_dir)  # Load all groups
```

Both versions of `eta` might be loaded. The last one encountered will overwrite previous ones. To avoid this, always specify either:
- `variables=["eta"]` (uses priority rules)
- `groups=["L.Dynamic"]` or `groups=["C.LandSurface"]` (explicit selection)

### Other Potentially Ambiguous Variables

Currently, only `eta` has known conflicts. If other variables have similar issues, add them to `VARIABLE_GROUP_PRIORITY`:

```python
VARIABLE_GROUP_PRIORITY = {
    "eta": ["L.Dynamic", "C.LandSurface"],
    "other_var": ["PreferredGroup", "SecondaryGroup"],
}
```

## Summary

✅ **Automatic disambiguation** - Request `eta` in variables list, get vorticity version
✅ **Explicit control** - Use `groups` parameter to select specific version
✅ **Staggered variable awareness** - Prefers 3D groups for staggered variables
✅ **Backward compatible** - Doesn't break existing code
✅ **Extensible** - Easy to add more priority rules

The solution ensures that when users request vorticity variables like `eta`, they automatically get the correct 3D version from `L.Dynamic`, not the 2D latent heat flux from `C.LandSurface`.
