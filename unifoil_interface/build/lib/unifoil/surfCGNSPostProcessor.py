import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Optional: keep if you're using niceplots globally; safe to remove otherwise
try:
    import niceplots
    plt.style.use(niceplots.get_style())
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass


class surfCGNSPostProcessor:
    """
    Cell-centered, z-plane-consistent CGNS post-processor.

    Key choices:
    - All fields are read at CELL CENTERS (coords = block.cell_centers().points).
    - z-plane is chosen automatically (dominant z among cell centers) per plot/extract.
    - Works for any field (scalar or vector). For vectors, choose magnitude/components via vel_component.
    - You can target a specific block with `block_index` (e.g., 2 for your main plane).
    """

    def __init__(self, cgns_file, airfoil_file=None):
        self.cgns_path = os.path.abspath(cgns_file)
        self.airfoil_path = os.path.abspath(airfoil_file) if airfoil_file else None

        if not os.path.exists(self.cgns_path):
            raise FileNotFoundError(f"CGNS file not found at: {self.cgns_path}")
        if self.airfoil_path and not os.path.exists(self.airfoil_path):
            raise FileNotFoundError(f"Airfoil file not found: {self.airfoil_path}")

        self.mesh = pv.read(self.cgns_path)
        self.x_af, self.y_af = None, None

        if self.airfoil_path:
            try:
                data = np.loadtxt(self.airfoil_path)
                # assumes columns: idx x y
                self.x_af = data[:, 1]
                self.y_af = data[:, 2]
            except Exception as e:
                print(f"Error reading airfoil file: {e}")

    # ----------------------------
    # Structure helpers
    # ----------------------------
    def display_structure(self):
        def explore_block(block, indent=0):
            prefix = "  " * indent
            if isinstance(block, pv.MultiBlock):
                print(f"{prefix}MultiBlock with {len(block)} blocks")
                for i, subblock in enumerate(block):
                    if subblock is None:
                        print(f"{prefix}  Block {i}: None")
                    else:
                        print(f"{prefix}  Block {i}:")
                        explore_block(subblock, indent + 2)
            else:
                print(f"{prefix}Type: {type(block)}")
                print(f"{prefix}Bounds: {block.bounds}")
                print(f"{prefix}Number of points: {block.n_points}")
                print(f"{prefix}Number of cells: {block.n_cells}")
                if block.point_data:
                    print(f"{prefix}Point data arrays:")
                    for k in block.point_data.keys():
                        arr = block.point_data[k]
                        print(f"{prefix}  - {k}: {arr.shape} {arr.dtype}")
                else:
                    print(f"{prefix}No point data arrays.")
                if block.cell_data:
                    print(f"{prefix}Cell data arrays:")
                    for k in block.cell_data.keys():
                        arr = block.cell_data[k]
                        print(f"{prefix}  - {k}: {arr.shape} {arr.dtype}")
                else:
                    print(f"{prefix}No cell data arrays.")

        print("\n=== Top-level CGNS Mesh Info ===")
        print(self.mesh)
        print("\n=== Exploring contents recursively ===")
        explore_block(self.mesh)

    def get_block_with_array(self, array_name=None, index=0):
        """
        Find the Nth dataset in the multiblock tree that contains `array_name`
        (in cell_data or point_data). If array_name is None, returns the Nth dataset.
        """
        found = []

        def _search(obj):
            if isinstance(obj, pv.MultiBlock):
                for sub in obj:
                    if sub is not None:
                        _search(sub)
            elif isinstance(obj, pv.DataSet):
                names = set()
                if obj.cell_data:
                    names |= set(obj.cell_data.keys())
                if obj.point_data:
                    names |= set(obj.point_data.keys())
                if array_name is None or array_name in names:
                    found.append(obj)

        _search(self.mesh)
        return found[index] if index < len(found) else None

    def available_fields(self, block_index=0):
        """List all array names in the chosen block (cell & point data)."""
        blk = self.get_block_with_array(index=block_index)
        if blk is None:
            return []
        names = set()
        if blk.cell_data:
            names |= set(blk.cell_data.keys())
        if blk.point_data:
            names |= set(blk.point_data.keys())
        return sorted(names)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    @staticmethod
    def _vec_component(data, comp):
        """
        From vector data array shape (N,3), return:
          'a' -> magnitude, 'b' -> x, 'c' -> y, 'd' -> z
        """
        if comp == 'a':
            return np.linalg.norm(data, axis=1)
        elif comp == 'b':
            return data[:, 0]
        elif comp == 'c':
            return data[:, 1]
        elif comp == 'd':
            return data[:, 2]
        else:
            raise ValueError("Invalid velocity component. Use 'a' (mag), 'b' (x), 'c' (y), or 'd' (z).")

    @staticmethod
    def _label_for(name, is_vector=False, comp='a'):
        lname = name.lower()
        if lname in ["mach", "ma"]:
            return r"$Ma$"
        if lname in ["coefpressure", "cp", "pressurecoefficient"]:
            return r"$C_p$"
        if lname in ["density", "rho"]:
            return r"$\rho$"
        if lname in ["pressure", "p"]:
            return "P"
        if lname in ["temperature", "t"]:
            return "T"
        if is_vector:
            return {
                'a': r"$|\mathbf{u}|$",
                'b': r"$u_x$",
                'c': r"$u_y$",
                'd': r"$u_z$",
            }.get(comp, name)
        return name

    @staticmethod
    def _dominant_z(coords):
        z_unique, counts = np.unique(np.round(coords[:, 2], 9), return_counts=True)
        return z_unique[np.argmax(counts)]

    def _ensure_cell_data(self, block, field_name):
        """
        Returns (coords, values) where coords are cell centers and values are
        the field values at cells. Converts point->cell if needed.
        """
        if field_name in block.cell_data:
            coords = block.cell_centers().points
            values = block.cell_data[field_name]
            return coords, values
        if field_name in block.point_data:
            tmp = block.point_data_to_cell_data(pass_point_data=False)
            coords = tmp.cell_centers().points
            values = tmp.cell_data[field_name]
            return coords, values
        raise ValueError(f"Field '{field_name}' not found in block (cell or point data).")

    # ----------------------------
    # Public API
    # ----------------------------
    def extract_xy_quantity(self, field_name, vel_component='a', block_index=0, save_path=None):
        """
        Return x, y, q on the dominant z-plane for the chosen block.
        - Scalars: q is the scalar.
        - Vectors: choose 'a' (|.|), 'b' (x), 'c' (y), 'd' (z).
        """
        block = self.get_block_with_array(array_name=field_name, index=block_index)
        if block is None:
            raise ValueError(f"No block found containing array: {field_name}")

        coords, raw = self._ensure_cell_data(block, field_name)
        z_plane = self._dominant_z(coords)
        z_mask = np.isclose(coords[:, 2], z_plane)

        x = coords[z_mask, 0]
        y = coords[z_mask, 1]

        if raw.ndim == 2 and raw.shape[1] == 3:
            q = self._vec_component(raw[z_mask], vel_component)
        else:
            q = raw[z_mask]

        if save_path is not None:
            np.savez(save_path, x=x, y=y, quantity=q, field=field_name, z_plane=z_plane)
            print(f"Saved extracted data to: {save_path}")

        return x, y, q

    def overlay_airfoil(self):
        if self.x_af is not None and self.y_af is not None:
            ax = plt.gca()
            ax.plot(self.x_af, self.y_af, color='white', linewidth=1.5, zorder=10)
            patch = patches.Polygon(np.c_[self.x_af, self.y_af], closed=True,
                                     facecolor='white', edgecolor='white', zorder=5)
            ax.add_patch(patch)

    def plot_field(self, field_name, block_index=0, vel_component='a',
                   xlim=(-0.1, 1.1), ylim=(-0.5, 0.5),
                   levels=200, cmap='viridis', overlay_airfoil=True):
        """
        Tricontour plot of a field on the dominant z-plane for the chosen block.
        """
        # Extract
        block = self.get_block_with_array(array_name=field_name, index=block_index)
        if block is None:
            print(f"No block found containing array: {field_name}")
            return

        coords, raw = self._ensure_cell_data(block, field_name)
        z_plane = self._dominant_z(coords)
        z_mask = np.isclose(coords[:, 2], z_plane)

        x = coords[z_mask, 0]
        y = coords[z_mask, 1]
        data = raw[z_mask]

        # Vector vs scalar
        is_vector = (data.ndim == 2 and data.shape[1] == 3)
        if is_vector:
            if vel_component == 'a':
                data = np.linalg.norm(data, axis=1)
                label, title = r"$|\mathbf{u}|$", f"{field_name} magnitude @ z={z_plane:g}"
            elif vel_component == 'b':
                data = data[:, 0]; label, title = r"$u_x$", f"{field_name}[x] @ z={z_plane:g}"
            elif vel_component == 'c':
                data = data[:, 1]; label, title = r"$u_y$", f"{field_name}[y] @ z={z_plane:g}"
            elif vel_component == 'd':
                data = data[:, 2]; label, title = r"$u_z$", f"{field_name}[z] @ z={z_plane:g}"
            else:
                raise ValueError("Invalid velocity component. Use 'a','b','c','d'.")
        else:
            label = self._label_for(field_name, is_vector=False)
            title = f"{field_name} @ z={z_plane:g}"

        # Plot (avoid tight_layout to dodge colorbar layout engine conflicts)
        fig, ax = plt.subplots(figsize=(8, 6))
        tri = ax.tricontourf(x, y, data, levels=levels, cmap=cmap)
        cb = fig.colorbar(tri, ax=ax, pad=0.02)
        cb.set_label(label, rotation=0, labelpad=20)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$", rotation=0)
        ax.set_title(title)
        # ax.set_aspect('equal', adjustable='box')  # enable if you want metric aspect
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if overlay_airfoil:
            self.overlay_airfoil()

        plt.show()
