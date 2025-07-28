import numpy as np
import re
import matplotlib.pyplot as plt
import niceplots
import os
from matplotlib.patches import Polygon

plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "Times New Roman"

def parse_zones_by_frequency(filename):
    with open(filename, 'r') as f:
        all_lines = f.readlines()

    for i, line in enumerate(all_lines):
        if 'enveloped' in line.lower():
            all_lines = all_lines[:i]
            break

    data_dict = {}
    current_freq = None
    current_data = []

    freq_pattern = re.compile(r'(\d+\.\d+)hz')
    all_lines = all_lines[7:]

    for line in all_lines:
        line = line.strip()

        if line.startswith('zone'):
            if current_freq is not None and current_data:
                data_array = np.array(current_data, dtype=np.float64)
                data_dict.setdefault(current_freq, []).append(data_array)
                current_data = []

            match = freq_pattern.search(line)
            if match:
                current_freq = float(match.group(1))

        elif line:
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 5:
                    current_data.append(values)
            except ValueError:
                continue

    if current_freq is not None and current_data:
        data_array = np.array(current_data, dtype=np.float64)
        data_dict.setdefault(current_freq, []).append(data_array)

    for freq in data_dict:
        data_dict[freq] = np.vstack(data_dict[freq])

    return data_dict

def get_file_path(default_path, use_all_defaults, label):
    if use_all_defaults:
        return default_path
    else:
        use = input(f"Use default for {label}? (y/n): ").strip().lower()
        if use == 'y':
            return default_path
        else:
            return input(f"Enter path to {label} file: ").strip()

def plot_nfactor(use_all_defaults):
    default_file = './input/nfactor_ts.dat'
    filename = get_file_path(default_file, use_all_defaults, "N-factor")

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    zone_data = parse_zones_by_frequency(filename)

    plt.figure(figsize=(7, 5))
    for i, (freq, data) in enumerate(sorted(zone_data.items())):
        if i >= 9:
            break
        x_c = data[:, 0]
        n_factor = data[:, 4]
        plt.plot(x_c, n_factor, label=f"{freq:.1f} Hz")

    plt.xlabel("x/c", fontsize=16)
    plt.ylabel("$N_{TS}$", fontsize=16, rotation=0, labelpad=20)
    plt.xlim([0, 0.83])
    plt.ylim([-0.5, 15])
    plt.legend(
        title="Frequency",
        title_fontsize=16,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(0.75, 0.8),
        borderaxespad=0,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.tick_params(axis='both', labelsize=16)
    plt.show()

def plot_airfoil(use_all_defaults):
    default_airfoil = './input/airfoil_coords.dat'
    default_transiloc = './input/transiLoc.dat'

    airfoil_file = get_file_path(default_airfoil, use_all_defaults, "airfoil coordinates")
    transiloc_file = get_file_path(default_transiloc, use_all_defaults, "transition locations")

    if not os.path.exists(airfoil_file):
        print(f"File not found: {airfoil_file}")
        return
    if not os.path.exists(transiloc_file):
        print(f"File not found: {transiloc_file}")
        return

    # Load airfoil coordinates
    data = np.loadtxt(airfoil_file)
    x = data[:, 1]
    y = data[:, 2]

    # Load transition locations
    transi_data = np.loadtxt(transiloc_file, usecols=(0, 1))
    ps_point = transi_data[0]  # Pressure side (red)
    ss_point = transi_data[1]  # Suction side (blue)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, color='black', linewidth=2)
    patch = Polygon(np.c_[x, y], closed=True, facecolor='lightgray', edgecolor='black', alpha=0.3)
    ax.add_patch(patch)

    # Overlay points
    ax.plot(*ps_point, 'ro', label="Suction side")
    ax.plot(*ss_point, 'bo', label="Pressure side")

    ax.set_aspect('equal')
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14, rotation=0)
    ax.grid(True)
    
    # Add padding to axis limits
    x_pad = 0.05 * (x.max() - x.min())
    y_pad = 0.5 * (y.max() - y.min())
    ax.set_xlim([x.min() - x_pad, x.max() + x_pad])
    ax.set_ylim([y.min() - y_pad, y.max() + y_pad])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.title("Transition locations")
    plt.tight_layout()
    plt.show()

# === ENTRY POINT ===
if __name__ == '__main__':
    print("\nSelect an option:")
    print("1. Plot N-factor (TS)")
    print("2. Plot airfoil shape with transition overlay")
    choice = input("Enter 1 or 2: ").strip()

    print("\nUse default input files for all? (y/n)")
    use_all_defaults = input("Your choice: ").strip().lower() == 'y'

    if choice == '1':
        plot_nfactor(use_all_defaults)
    elif choice == '2':
        plot_airfoil(use_all_defaults)
    else:
        print("Invalid choice. Exiting.")
