import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import niceplots

plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "Times New Roman"

class sup_transi:
    def __init__(self, nfactor_file='./input/nfactor_ts.dat',
                 airfoil_file='./input/airfoil_coords.dat',
                 transiloc_file='./input/transiLoc.dat'):
        self.nfactor_file = nfactor_file
        self.airfoil_file = airfoil_file
        self.transiloc_file = transiloc_file

    def parse_zones_by_frequency(self):
        if not os.path.exists(self.nfactor_file):
            raise FileNotFoundError(f"N-factor file not found: {self.nfactor_file}")

        with open(self.nfactor_file, 'r') as f:
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

    def plot_nfactor(self, save_npz_path=None):
        zone_data = self.parse_zones_by_frequency()

        plt.figure(figsize=(7, 5))
        npz_dict = {}

        for i, (freq, data) in enumerate(sorted(zone_data.items())):
            if i >= 9:
                break
            x_c = data[:, 0]
            n_factor = data[:, 4]
            plt.plot(x_c, n_factor, label=f"{freq:.1f} Hz")
            npz_dict[f"x_c_{freq:.1f}Hz"] = x_c
            npz_dict[f"N_{freq:.1f}Hz"] = n_factor

        if save_npz_path is not None:
            np.savez(save_npz_path, **npz_dict)
            print(f"N-factor data saved to: {save_npz_path}")

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

    def plot_airfoil_with_transition(self, save_npz_path=None):
        if not os.path.exists(self.airfoil_file):
            raise FileNotFoundError(f"Airfoil file not found: {self.airfoil_file}")
        if not os.path.exists(self.transiloc_file):
            raise FileNotFoundError(f"Transition location file not found: {self.transiloc_file}")

        data = np.loadtxt(self.airfoil_file)
        x = data[:, 1]
        y = data[:, 2]

        transi_data = np.loadtxt(self.transiloc_file, usecols=(0, 1))
        ps_point = transi_data[1]  # Pressure side (red)
        ss_point = transi_data[0]  # Suction side (blue)

        if save_npz_path is not None:
            np.savez(
                save_npz_path,
                x=x,
                y=y,
                ps_point=ps_point,
                ss_point=ss_point
            )
            print(f"Airfoil and transition data saved to: {save_npz_path}")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(x, y, color='black', linewidth=2)
        patch = Polygon(np.c_[x, y], closed=True, facecolor='lightgray', edgecolor='black', alpha=0.3)
        ax.add_patch(patch)

        ax.plot(*ps_point, 'ro', label="Pressure side")
        ax.plot(*ss_point, 'bo', label="Suction side")

        ax.set_aspect('equal')
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14, rotation=0)
        ax.grid(False)

        x_pad = 0.05 * (x.max() - x.min())
        y_pad = 0.5 * (y.max() - y.min())
        ax.set_xlim([x.min() - x_pad, x.max() + x_pad])
        ax.set_ylim([y.min() - y_pad, y.max() + y_pad])
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        plt.title("Transition locations")
        plt.tight_layout()
        plt.show()
    
    def get_data(self):
        """
        Returns parsed N-factor and transition data without plotting.
    
        Returns
        -------
        tuple:
            (nfactor_data, transi_data)
            - nfactor_data: dict {frequency (Hz): ndarray}
            - transi_data: dict with keys ('x', 'y', 'ps_point', 'ss_point')
        """
        nfactor_data = None
        transi_data = None
    
        # ---------- N-Factor ----------
        if os.path.exists(self.nfactor_file):
            try:
                nfactor_data = self.parse_zones_by_frequency()
            except Exception as e:
                print(f"[sup_transi] ⚠️ Error parsing N-factor data: {e}")
        else:
            print(f"[sup_transi] ⚠️ Missing N-factor file: {self.nfactor_file}")
    
        # ---------- Transition + Geometry ----------
        if os.path.exists(self.airfoil_file) and os.path.exists(self.transiloc_file):
            try:
                airfoil = np.loadtxt(self.airfoil_file)
                x, y = airfoil[:, 1], airfoil[:, 2]
                transi = np.loadtxt(self.transiloc_file, usecols=(0, 1))
                ps_point, ss_point = transi[1], transi[0]
                transi_data = {
                    "x": x,
                    "y": y,
                    "ps_point": ps_point,
                    "ss_point": ss_point,
                }
            except Exception as e:
                print(f"[sup_transi] ⚠️ Error loading transition geometry: {e}")
        else:
            print(f"[sup_transi] ⚠️ Missing transition or airfoil file.")
    
        return nfactor_data, transi_data
