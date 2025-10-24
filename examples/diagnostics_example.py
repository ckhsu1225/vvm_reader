"""
Example: Computing Diagnostic Variables with VVM Reader

This example demonstrates how to use the diagnostics module to compute
derived atmospheric variables from VVM model output.
"""

import vvm_reader as vvm

# ============================================================================
# Example 1: List Available Diagnostic Variables
# ============================================================================

print("="*70)
print("Example 1: List Available Diagnostic Variables")
print("="*70)

available = vvm.list_available_diagnostics()
print(f"\nAvailable diagnostic variables ({len(available)}):")
for var in available:
    print(f"  - {var}")

# ============================================================================
# Example 2: Get Metadata for Specific Diagnostic Variable
# ============================================================================

print("\n" + "="*70)
print("Example 2: Get Metadata for Diagnostic Variables")
print("="*70)

from vvm_reader.diagnostics import get_diagnostic_metadata

for var_name in ['T', 'RH', 'MSE', 'CWV']:
    metadata = get_diagnostic_metadata(var_name)
    print(f"\n{var_name}:")
    print(f"  Long name: {metadata['long_name']}")
    print(f"  Units: {metadata['units']}")
    print(f"  Description: {metadata['description']}")

# ============================================================================
# Example 3: Understand Variable Dependencies
# ============================================================================

print("\n" + "="*70)
print("Example 3: Understanding Variable Dependencies")
print("="*70)

from vvm_reader.diagnostics import (
    get_required_file_variables,
    get_required_profiles
)

diagnostic_vars = ['T', 'RH', 'MSE']

print(f"\nTo compute {diagnostic_vars}:")
print(f"  Required file variables: {get_required_file_variables(diagnostic_vars)}")
print(f"  Required fort.98 profiles: {get_required_profiles(diagnostic_vars)}")

# ============================================================================
# Example 4: Automatic Diagnostic Computation (Recommended)
# ============================================================================

print("\n" + "="*70)
print("Example 4: Automatic Diagnostic Computation (Recommended)")
print("="*70)

print("\nThe simplest way: mix file and diagnostic variables in one call:")
print("""
sim_dir = "/path/to/simulation"

# Automatically computes T, RH, MSE from th and qv
ds = vvm.open_vvm_dataset(
    sim_dir,
    variables=["th", "qv", "T", "RH", "MSE"]
)

# T, RH, MSE are automatically computed!
temperature = ds['T']
relative_humidity = ds['RH']
moist_static_energy = ds['MSE']
""")

# ============================================================================
# Example 5: Manual Diagnostic Computation (Advanced)
# ============================================================================

print("\n" + "="*70)
print("Example 5: Manual Diagnostic Computation (Advanced)")
print("="*70)

print("\nFor more control, disable auto-computation:")
print("""
# Step 1: Load model output only (no auto-computation)
sim_dir = "/path/to/simulation"
ds = vvm.open_vvm_dataset(
    sim_dir,
    variables=["th", "qv", "u", "v"],
    auto_compute_diagnostics=False  # Disable automatic computation
)

# Step 2: Manually compute diagnostic variables when needed
ds = vvm.compute_diagnostics(ds, ["T", "RH"], sim_dir)

# Step 3: Later, compute more diagnostics
ds = vvm.compute_diagnostics(ds, ["MSE", "CWV"], sim_dir)

# Useful for:
# - Computing different diagnostics at different stages
# - Debugging diagnostic computation issues
# - Performance optimization (compute only when needed)
""")

# ============================================================================
# Example 6: Print Full Diagnostic Info
# ============================================================================

print("\n" + "="*70)
print("Example 6: Full Diagnostic Variable Information")
print("="*70)

vvm.diagnostics.print_diagnostic_info()

# ============================================================================
# Example 7: Inspect Variable Dependencies (Advanced)
# ============================================================================

print("\n" + "="*70)
print("Example 7: Inspect Dependency Tree")
print("="*70)

from vvm_reader.diagnostics import get_registry

registry = get_registry()

print("\nDependency details for selected variables:")
for var_name in ['T', 'RH', 'theta_e', 'MSE']:
    diag_var = registry.get(var_name)
    print(f"\n{var_name}:")
    print(f"  File dependencies: {diag_var.file_dependencies}")
    print(f"  Profile dependencies: {diag_var.profile_dependencies}")
    print(f"  Diagnostic dependencies: {diag_var.diagnostic_dependencies}")

# ============================================================================
# Notes on Usage
# ============================================================================

print("\n" + "="*70)
print("IMPORTANT NOTES")
print("="*70)

print("""
1. Pressure Approximation:
   - All diagnostics use PIBAR (background Exner function) from fort.98
   - This neglects pressure perturbations (π')
   - Typical errors: <2% for most applications
   - Appropriate for: weak to moderate convection
   - May be less accurate for: extreme convection (e.g., supercells)

2. CAPE/CIN Implementation:
   - Currently placeholder implementations (return zeros)
   - Full parcel theory implementation planned for future release
   - For production CAPE/CIN, consider other libraries (e.g., MetPy)

3. Required fort.98:
   - Most diagnostics require reference profiles from fort.98
   - Ensure fort.98 exists in your simulation directory
   - Profiles used: PIBAR, PBAR, RHO, THBAR, QVBAR

4. Performance:
   - Dependency resolution is automatic
   - Variables computed in optimal order
   - Fort.98 reading is cached (LRU cache)
""")

print("\n" + "="*70)
print("Example Complete!")
print("="*70)
