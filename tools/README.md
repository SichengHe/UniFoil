## post_processor_class_transi usage guide

This class provides utilities to:
- Parse and plot N-factor data from a `.dat` file
- Plot the airfoil shape overlaid with transition locations
- Optionally save extracted data to `.npz` format

📁 Folder Structure (Example)

```
UniFoil/
├── input/
│   ├── nfactor_ts.dat
│   ├── airfoil_coords.dat
│   └── transiLoc.dat
├── tools/
│   ├── __init__.py
│   └── nfactor_plotter.py      # ← Contains NFactorPlotter class
├── examples/
│   └── visualize_nfactor.py    # ← Your runner script
```

Example: Basic Usage

```python
# examples/visualize_nfactor.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.nfactor_plotter import NFactorPlotter

# Initialize with default or custom paths
plotter = NFactorPlotter(
    nfactor_file="./input/nfactor_ts.dat",
    airfoil_file="./input/airfoil_coords.dat",
    transiloc_file="./input/transiLoc.dat"
)

# Plot N-factor curves (and save parsed data)
plotter.plot_nfactor(save_npz_path="output/nfactor_curves.npz")

# Plot airfoil with transition overlay (and save data)
plotter.plot_airfoil_with_transition(save_npz_path="output/airfoil_transition.npz")
```

Saved Data Format (.npz)

- `nfactor_curves.npz` contains:
  - `x_c_150.0Hz`, `N_150.0Hz`, etc. for each frequency

- `airfoil_transition.npz` contains:
  - `x`, `y` → airfoil coordinates  
  - `ps_point`, `ss_point` → transition points on suction and pressure sides
  
## post_processor_class_turb usage guide
The `CGNSPostProcessor` class provides an interface to:

- Load a CGNS file and inspect its structure
- Plot scalar/vector fields (like Mach, Pressure, Velocity)
- Overlay the airfoil shape (optional)
- Save 2D field slices (x, y, quantity) to `.npz`

---

## 📁 Folder Structure Example

```
UniFoil/
├── input/
│   ├── airfoil_incomp.cgns
│   └── airfoil_coords.dat
├── tools/
│   ├── __init__.py
│   └── post_processor_class.py   # ← Contains CGNSPostProcessor
├── examples/
│   └── transi_data_visualize/
│       ├── runner_script.py      # ← Your main script
```

---

## 🚀 Example Usage in `runner_script.py`

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.post_processor_class import CGNSPostProcessor

# Initialize the postprocessor
post = CGNSPostProcessor(
    cgns_file="airfoil_incomp.cgns",
    airfoil_file="airfoil_coords.dat"
)

# Display the CGNS tree structure
post.display_structure()

# Plot a scalar or vector field
post.plot_field(field_name="Velocity", block_index=2, vel_component='a')
# Other examples: "Mach", "Pressure", "SkinFrictionMagnitude", etc.

# Extract (x, y, quantity) and save to .npz
x, y, q = post.extract_xy_quantity(
    field_name="Velocity",
    block_index=2,
    vel_component='a',
    save_path="output/velocity_slice_z0.npz"
)
```

---

## 🧠 Supported Field Names

- `"Velocity"` → use `vel_component='a'`, `'b'`, or `'c'` for magnitude, x, or y components
- `"Mach"`, `"Pressure"`, `"Temperature"`, `"Density"`, `"CoefPressure"`
- `"SkinFrictionMagnitude"`, `"SkinFrictionX"`, `"YPlus"` (opens 3D PyVista viewer)

---

## 💾 Saved `.npz` Contents

When using `extract_xy_quantity(..., save_path="file.npz")`, the saved file will contain:

- `x`: 1D array of x coordinates at z = 0
- `y`: 1D array of y coordinates at z = 0
- `quantity`: extracted scalar field (or velocity component/magnitude)