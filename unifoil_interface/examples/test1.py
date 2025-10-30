from unifoil.extract_data import ExtractData
import matplotlib.pyplot as plt

# ===========================================
#   Initialize the ExtractData class
# ===========================================
ed = ExtractData()

# Specify which airfoil and case to work with
airfoil_number = 5
case_number = 2

# ===========================================
# 1️⃣ Display CGNS structure hierarchy
# ===========================================
print("\n=== [1] Display CGNS File Structure ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    action="display_structure"
)

# ===========================================
# 2️⃣ Plot scalar fields (Pressure, Mach, etc.)
# ===========================================
print("\n=== [2] Plot Coefficient of Pressure field ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="CoefPressure",
    action="plot_field",
    block_index=2
)

print("\n=== [3] Plot Mach field ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="Mach",
    action="plot_field",
    block_index=2
)

# ===========================================
# 3️⃣ Plot velocity components and magnitude
# ===========================================
print("\n=== [4] Plot Velocity Magnitude (|u|) ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="Velocity",
    vel_component='a',  # |u|
    action="plot_field"
)

print("\n=== [5] Plot Velocity X-component (u_x) ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="Velocity",
    vel_component='b',  # u_x
    action="plot_field"
)

print("\n=== [6] Plot Velocity Y-component (u_y) ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="Velocity",
    vel_component='c',  # u_y
    action="plot_field"
)

# ===========================================
# 4️⃣ Extract field values (x, y, q) at z=0
# ===========================================
print("\n=== [7] Extract Cp field at z = 0 and save ===")
result = ed.surf_turb(
    airfoil_number=airfoil_number,
    case_number=case_number,
    field_name="CoefPressure",
    action="extract_xy_quantity",
    block_index=2,
    save_path=f"pressure_data_airfoil{airfoil_number}_case{case_number}.npz"
)

if result:
    x, y, q = result
    print(f"Extracted {len(x)} points. Example values:")
    print("x[0:5] =", x[:5])
    print("y[0:5] =", y[:5])
    print("q[0:5] =", q[:5])

    # Quick scatter plot of extracted data
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, y, c=q, cmap="coolwarm")
    plt.colorbar(sc, label="$C_p$")
    plt.xlabel("x")
    plt.ylabel("y", rotation=0)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(f"Extracted $C_p$ field (Airfoil {airfoil_number}, Case {case_number})")
    plt.show()


print("\n✅ All demo steps completed successfully.")
