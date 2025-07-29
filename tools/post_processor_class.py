import pyvista as pv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import niceplots
plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "Times New Roman"

class CGNSPostProcessor:
    def __init__(self, cgns_file, airfoil_file=None):
        self.cgns_path = os.path.join("input", cgns_file)
        self.airfoil_path = os.path.join("input", airfoil_file) if airfoil_file else None
        self.mesh = None
        self.x_af, self.y_af = None, None

        if not os.path.exists(self.cgns_path):
            raise FileNotFoundError(f"CGNS file not found at: {self.cgns_path}")

        if self.airfoil_path and not os.path.exists(self.airfoil_path):
            raise FileNotFoundError(f"Airfoil file not found: {self.airfoil_path}")

        self._load_data()
    
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
                print(f"{prefix}Array names: {block.array_names}")

        print("\n=== Top-level CGNS Mesh Info ===")
        print(self.mesh)
        print("\n=== Exploring contents recursively ===")
        explore_block(self.mesh)


    def _load_data(self):
        self.mesh = pv.read(self.cgns_path)
        print("CGNS file loaded successfully.")
        if self.airfoil_path:
            try:
                data = np.loadtxt(self.airfoil_path)
                self.x_af = data[:, 1]
                self.y_af = data[:, 2]
            except Exception as e:
                print(f"Error reading airfoil file: {e}")

    def get_block_with_array(self, array_name="Velocity", index=0):
        found_blocks = []
        def _search_recursive(obj):
            if isinstance(obj, pv.MultiBlock):
                for sub in obj:
                    _search_recursive(sub)
            elif isinstance(obj, pv.DataSet):
                if array_name in obj.array_names or array_name in obj.cell_data:
                    found_blocks.append(obj)
        _search_recursive(self.mesh)
        return found_blocks[index] if index < len(found_blocks) else None

    def overlay_airfoil(self):
        if self.x_af is not None and self.y_af is not None:
            plt.plot(self.x_af, self.y_af, color='white', linewidth=1.5, zorder=10)
            patch = patches.Polygon(np.c_[self.x_af, self.y_af], closed=True,
                                     facecolor='white', edgecolor='white', zorder=5)
            plt.gca().add_patch(patch)

    def plot_field(self, field_name, block_index=2, vel_component='a'):
        block = self.get_block_with_array(array_name=field_name, index=block_index)
        if block is None:
            print(f"No block found containing array: {field_name}")
            return

        if field_name in ["SkinFrictionMagnitude", "SkinFrictionX", "YPlus"]:
            print(f"Launching 3D PyVista plot for '{field_name}'...")
            while isinstance(self.mesh, pv.MultiBlock):
                self.mesh = self.mesh[0]
            plotter = pv.Plotter()
            plotter.add_mesh(self.mesh, scalars=field_name, cmap="coolwarm", show_edges=False)
            plotter.add_axes()
            plotter.show()
            return

        if field_name == "Mach":
            cell_centers = block.cell_centers()
            coords = cell_centers.points
            quantity = block.cell_data[field_name]
        elif field_name == "Velocity":
            if field_name in block.cell_data:
                block = block.cell_data_to_point_data()
            coords = block.points
            quantity = block[field_name]
        else:
            coords = block.cell_centers().points if field_name in block.cell_data else block.points
            quantity = block.cell_data[field_name] if field_name in block.cell_data else block[field_name]

        z0_mask = np.isclose(coords[:, 2], 0.0)
        x, y = coords[z0_mask, 0], coords[z0_mask, 1]
        data = quantity[z0_mask]

        if field_name == "Velocity":
            if vel_component == "a":
                data = np.linalg.norm(data, axis=1)
                label, title = "$|u|$", "Velocity Magnitude at z = 0"
            elif vel_component == "b":
                data = data[:, 0]
                label, title = "$u_x$", "Velocity X-Component at z = 0"
            elif vel_component == "c":
                data = data[:, 1]
                label, title = "$u_y$", "Velocity Y-Component at z = 0"
            else:
                raise ValueError("Invalid velocity component. Use 'a', 'b', or 'c'.")
        else:
            if field_name=='Mach':
                label = "$Ma$"
 
            elif field_name=='Density':
                label = "$\rho$"

            elif field_name=='Pressure':
                label = "P"

            elif field_name=='Temperature':
                label = "T"

            elif field_name=='CoefPressure':
                label = "$C_p$"

            title =f"{field_name} field"

        plt.figure(figsize=(8, 6))
        tri = plt.tricontourf(x, y, data, levels=200, cmap='viridis')
        cbar = plt.colorbar(tri)
        cbar.set_label(label, rotation=0, labelpad=20)
        plt.xlabel("$x$")
        plt.ylabel("$y$", rotation=0)
        plt.title(title)
        plt.axis("equal")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.5, 0.5])
        #plt.tight_layout()
        self.overlay_airfoil()
        plt.show()
    
    def extract_xy_quantity(self, field_name, block_index=2, vel_component='a', save_path=None):
        """
        Returns x, y, and selected quantity at z=0 for the given field.
        
        Parameters:
        -----------
        field_name : str
            Name of the field to extract (e.g., 'Velocity', 'Pressure').
        block_index : int
            Index of the block containing the field.
        vel_component : str
            'a' for |u|, 'b' for u_x, 'c' for u_y (used only for 'Velocity').
    
        Returns:
        --------
        Tuple of numpy arrays: (x, y, quantity)
        """
        block = self.get_block_with_array(array_name=field_name, index=block_index)
        if block is None:
            raise ValueError(f"No block found containing array: {field_name}")
    
        if field_name == "Velocity":
            if field_name in block.cell_data:
                block = block.cell_data_to_point_data()
            coords = block.points
            field_data = block[field_name]
            z_vals = coords[:, 2]
            z0_mask = np.isclose(z_vals, 0.0)
            x, y = coords[z0_mask, 0], coords[z0_mask, 1]
            vel_xy = field_data[z0_mask]
    
            if vel_component == 'a':
                quantity = np.linalg.norm(vel_xy, axis=1)
            elif vel_component == 'b':
                quantity = vel_xy[:, 0]
            elif vel_component == 'c':
                quantity = vel_xy[:, 1]
            else:
                raise ValueError("Invalid velocity component: choose 'a', 'b', or 'c'.")
    
        else:
            if field_name in block.cell_data:
                coords = block.cell_centers().points
                quantity = block.cell_data[field_name]
            else:
                coords = block.points
                quantity = block[field_name]
    
            z_vals = coords[:, 2]
            z0_mask = np.isclose(z_vals, 0.0)
            x, y = coords[z0_mask, 0], coords[z0_mask, 1]
            quantity = quantity[z0_mask]
        if save_path is not None:
            np.savez(save_path, x=x, y=y, quantity=quantity)
            print(f"Saved extracted data to: {save_path}")
        return x, y, quantity
    
