'''
READ:
    This code contains post_process script to visualize data in the CGNS files.
'''

import pyvista as pv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def get_block_with_array(mesh, array_name="Velocity", index=0):
    found_blocks = []
    def _search_recursive(obj):
        if isinstance(obj, pv.MultiBlock):
            for sub in obj:
                _search_recursive(sub)
        elif isinstance(obj, pv.DataSet):
            if array_name in obj.array_names or array_name in obj.cell_data:
                found_blocks.append(obj)
    _search_recursive(mesh)
    if index < len(found_blocks):
        return found_blocks[index]
    else:
        print(f"No block found at index {index} with array '{array_name}' (found {len(found_blocks)} blocks).")
        return None

def get_airfoil_coords():
    default_datfile = "airfoil_coords.dat"
    use_default = input(f"Use default airfoil shape file '{default_datfile}'? (y/n): ").strip().lower()
    if use_default == 'y':
        datfile = default_datfile
    else:
        datfile = input("Enter .dat filename inside 'input/' folder (e.g., myfoil.dat): ").strip()
    dat_path = os.path.join("input", datfile)
    if not os.path.exists(dat_path):
        print(f"Airfoil coord file not found: {dat_path}")
        return None, None
    try:
        data = np.loadtxt(dat_path)
        x_af = data[:, 1]
        y_af = data[:, 2]
        return x_af, y_af
    except Exception as e:
        print(f"Error reading airfoil file: {e}")
        return None, None

def overlay_airfoil_on_plot(x_af, y_af):
    if x_af is not None and y_af is not None:
        plt.plot(x_af, y_af, color='white', linewidth=1.5, zorder=10)
        patch = patches.Polygon(np.c_[x_af, y_af], closed=True,
                                 facecolor='white', edgecolor='white', zorder=5)
        plt.gca().add_patch(patch)

# === MAIN ===

# --- Select CGNS file
default_filename = "airfoil_incomp.cgns"
use_default = input(f"Use default CGNS file '{default_filename}'? (y/n): ").strip().lower()
filename = default_filename if use_default == 'y' else input("Enter CGNS filename inside 'input/' folder (e.g., my_case.cgns): ").strip()
file_path = os.path.join("input", filename)
if not os.path.exists(file_path):
    print(f"File not found at: {file_path}")
    exit()

# --- Select airfoil coordinates file
x_af, y_af = get_airfoil_coords()

# --- Load CGNS
mesh = pv.read(file_path)
print("=== CGNS file loaded successfully ===\n")

choice = input("Do you want to display the CGNS file structure? (y/n): ").strip().lower()
if choice == 'y':
    print("\n=== Top-level CGNS contents ===")
    print(mesh)
    print("\n=== Exploring contents recursively ===")
    explore_block(mesh)
else:
    flag='cont'
    while flag=='cont':
        
        available_fields = ['Mach', 'Density', 'Pressure', 'Temperature', 'Velocity',
                            'CoefPressure', 'SkinFrictionMagnitude', 'YPlus', 'SkinFrictionX']
    
        print("\nSelect a field to extract:")
        for i, field in enumerate(available_fields):
            print(f"{i+1}. {field}")
    
        option = input("Enter number corresponding to field\nOr enter e exit: ").strip()
        if option=='e':
            break
        try:
            index = int(option) - 1
            field_name = available_fields[index]
        except (IndexError, ValueError):
            print("Invalid selection. Exiting.")
            exit()
    
        block = get_block_with_array(mesh, array_name=field_name, index=2)
    
        if block is not None:
            skin_friction_fields = ["SkinFrictionMagnitude", "SkinFrictionX", "YPlus"]
    
            # === SKIN FRICTION ===
            if field_name in skin_friction_fields:
                print(f"Launching 3D PyVista plot for '{field_name}'...")
                while isinstance(mesh, pv.MultiBlock):
                    mesh = mesh[0]
                plotter = pv.Plotter()
                plotter.add_mesh(mesh, scalars=field_name, cmap="coolwarm", show_edges=False)
                plotter.add_axes()
                plotter.show()
    
            # === MACH ===
            elif field_name == "Mach":
                print(f"Using cell-centered data for '{field_name}'...")
                cell_centers = block.cell_centers()
                coords = cell_centers.points
                quantity = block.cell_data[field_name]
                z_vals = coords[:, 2]
                z0_mask = np.isclose(z_vals, 0.0)
                x, y, q_z0 = coords[z0_mask, 0], coords[z0_mask, 1], quantity[z0_mask]
    
                plt.figure(figsize=(8, 6))
                tri = plt.tricontourf(x, y, q_z0, levels=200, cmap='viridis')
                cbar = plt.colorbar(tri)
                cbar.set_label("$Mach$", rotation=0, labelpad=20)
                plt.xlabel("$x$")
                plt.ylabel("$y$", rotation=0)
                plt.title("Mach (Cell-Centered) at z = 0")
                plt.axis("equal")
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.5, 0.5])
                plt.tight_layout()
                overlay_airfoil_on_plot(x_af, y_af)
                plt.show()
    
            # === VELOCITY ===
            elif field_name == "Velocity":
                if field_name in block.cell_data:
                    print("Converting velocity from cell_data to point_data...")
                    block = block.cell_data_to_point_data()
                coords = block.points
                field_data = block[field_name]
    
                print("\nSelect Velocity component to plot:")
                print("a. Magnitude (|u|)")
                print("b. x-component (u_x)")
                print("c. y-component (u_y)")
                vel_option = input("Enter choice (a/b/c): ").strip().lower()
    
                z_vals = coords[:, 2]
                z0_mask = np.isclose(z_vals, 0.0)
                x, y = coords[z0_mask, 0], coords[z0_mask, 1]
                vel_xy = field_data[z0_mask]
    
                if vel_option == "a":
                    quantity, label, title = np.linalg.norm(vel_xy, axis=1), "$|u|$", "Velocity Magnitude at z = 0"
                elif vel_option == "b":
                    quantity, label, title = vel_xy[:, 0], "$u_x$", "Velocity X-Component at z = 0"
                elif vel_option == "c":
                    quantity, label, title = vel_xy[:, 1], "$u_y$", "Velocity Y-Component at z = 0"
                else:
                    print("Invalid velocity component selection.")
                    exit()
    
                plt.figure(figsize=(8, 6))
                tri = plt.tricontourf(x, y, quantity, levels=200, cmap='viridis')
                cbar = plt.colorbar(tri)
                cbar.set_label(label, rotation=0, labelpad=20)
                plt.xlabel("$x$")
                plt.ylabel("$y$", rotation=0)
                plt.title(title)
                plt.axis("equal")
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.5, 0.5])
                plt.tight_layout()
                overlay_airfoil_on_plot(x_af, y_af)
                plt.show()
    
            # === OTHER FIELDS ===
            else:
                if field_name in block.cell_data:
                    print(f"Using cell-centered data for '{field_name}'...")
                    cell_centers = block.cell_centers()
                    coords = cell_centers.points
                    quantity = block.cell_data[field_name]
                else:
                    coords = block.points
                    quantity = block[field_name]
    
                z_vals = coords[:, 2]
                z0_mask = np.isclose(z_vals, 0.0)
                x, y, q_z0 = coords[z0_mask, 0], coords[z0_mask, 1], quantity[z0_mask]
    
                plt.figure(figsize=(8, 6))
                tri = plt.tricontourf(x, y, q_z0, levels=200, cmap='viridis')
                cbar = plt.colorbar(tri)
                cbar.set_label(f"${field_name}$", rotation=0, labelpad=20)
                plt.xlabel("$x$")
                plt.ylabel("$y$", rotation=0)
                plt.title(f"{field_name} at z = 0")
                plt.axis("equal")
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.5, 0.5])
                plt.tight_layout()
                overlay_airfoil_on_plot(x_af, y_af)
                plt.show()
        else:
            print(f"No block found containing array: {field_name}")
            flag=0
        
