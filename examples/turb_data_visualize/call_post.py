from post_processor_class import CGNSPostProcessor

# --- Static user input ---
cgns_filename = "airfoil_incomp.cgns"
airfoil_filename = "airfoil_coords.dat"
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
