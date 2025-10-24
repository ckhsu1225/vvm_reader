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
# Example 4: Compute Diagnostic Variables (Standalone)
# ============================================================================

print("\n" + "="*70)
print("Example 4: Compute Diagnostic Variables Standalone")
print("="*70)

print("\nThis example shows how to compute diagnostics from an existing dataset:")
print("""
# Step 1: Load model output
sim_dir = "/path/to/simulation"
ds = vvm.open_vvm_dataset(sim_dir, variables=["th", "qv", "u", "v"])

# Step 2: Compute diagnostic variables
ds_with_diagnostics = vvm.compute_diagnostics(
    ds,
    variables=["T", "RH", "MSE", "CWV"],
    sim_dir=sim_dir
)

# Step 3: Access computed variables
temperature = ds_with_diagnostics['T']
relative_humidity = ds_with_diagnostics['RH']
""")

# ============================================================================
# Example 5: Print Full Diagnostic Info
# ============================================================================

print("\n" + "="*70)
print("Example 5: Full Diagnostic Variable Information")
print("="*70)

vvm.diagnostics.print_diagnostic_info()

# ============================================================================
# Example 6: Inspect Variable Dependencies (Advanced)
# ============================================================================

print("\n" + "="*70)
print("Example 6: Inspect Dependency Tree")
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
