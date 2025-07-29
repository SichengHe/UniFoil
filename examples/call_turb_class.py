import os
import sys

# Get absolute path to the UniFoil root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add UniFoil to sys.path
sys.path.append(ROOT_DIR)
from tools.post_processor_class_turb import CGNSPostProcessor

# --- Static user input ---
cgns_filename = "./sample_input_data/airfoil_incomp.cgns"
airfoil_filename = "./sample_input_data/airfoil_coords.dat"
field_to_plot = "CoefPressure"            # e.g., "Mach", "SkinFrictionX", "Pressure", etc.
block_index = 2                       # Usually 2, but depends on file structure
vel_component = 'a'                  # 'a' for |u|, 'b' for u_x, 'c' for u_y (only for Velocity)

# --- Initialize and Plot ---
post = CGNSPostProcessor(cgns_filename, airfoil_file=airfoil_filename)
post.plot_field(field_name=field_to_plot, block_index=block_index, vel_component=vel_component)
post.display_structure()

x, y, q = post.extract_xy_quantity(
    field_name="Density", 
    block_index=2, 
    vel_component='a', 
    save_path="./velocity_mag_z0.npz")
