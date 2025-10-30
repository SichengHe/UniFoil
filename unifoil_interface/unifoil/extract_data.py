import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots
import pickle

from unifoil.surfCGNSPostProcessor import surfCGNSPostProcessor


plt.style.use(niceplots.get_style())

class ExtractData:
    """
    Class for extracting airfoil geometry and case information
    from the current working directory.

    Expected working directory contents:
      - Airfoil_Case_Data_turb.csv
      - Airfoil_Case_Data_Trans_Lam.csv
      - airfoil_ft_geom/ (contains airfoil_###.dat)
      - airfoil_nlf_geom/ (contains airfoil_###.dat)
    """

    def __init__(self):
        # Paths in the current working directory
        self.cwd = os.getcwd()
        self.ft_geom_path = os.path.join(self.cwd, "airfoil_ft_geom")
        self.nlf_geom_path = os.path.join(self.cwd, "airfoil_nlf_geom")

        # CSV paths
        self.turb_csv = os.path.join(self.cwd, "Airfoil_Case_Data_turb.csv")
        self.translam_csv = os.path.join(self.cwd, "Airfoil_Case_Data_Trans_Lam.csv")

        # Check folders
        if not os.path.exists(self.ft_geom_path):
            print(f"Warning: '{self.ft_geom_path}' not found.")
        if not os.path.exists(self.nlf_geom_path):
            print(f"Warning: '{self.nlf_geom_path}' not found.")

    # ----------------------------------------------------------------------
    def extract_airfoil_coords(self, airfoil_number, source="turb", plot_flag=False):
        """
        Extracts x and y coordinates for the airfoil corresponding to the given number.
    
        Parameters
        ----------
        airfoil_number : int
            The airfoil number (as in the 'airfoil' column of the CSV).
        source : str, optional
            Which dataset to use ("turb" or "translam").
        plot_flag : bool, optional
            Whether to plot the airfoil coordinates.
    
        Returns
        -------
        tuple of np.ndarray
            (x, y) coordinates of the specified airfoil.
        """
        # Select source file and geometry folder
        if source.lower() == "turb":
            csv_path = self.turb_csv
            geom_path = self.ft_geom_path
            ft_style = True
        else:
            csv_path = self.translam_csv
            geom_path = self.nlf_geom_path
            ft_style = False
    
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
        # Read CSV
        df = pd.read_csv(csv_path)
        if "airfoil" not in df.columns:
            raise ValueError("CSV must have an 'airfoil' column.")
    
        # Locate row
        row = df.loc[df["airfoil"] == airfoil_number]
        if row.empty:
            raise ValueError(f"Airfoil {airfoil_number} not found in {os.path.basename(csv_path)}")
    
        airfoil_idx = int(row["airfoil"].values[0])
    
        # Smart filename pattern based on dataset type
        if ft_style:
            # e.g., airfoil_1.dat, airfoil_2.dat
            dat_filename = os.path.join(geom_path, f"airfoil_{airfoil_idx}.dat")
        else:
            # e.g., airfoil_001.dat, airfoil_002.dat
            dat_filename = os.path.join(geom_path, f"airfoil_{airfoil_idx:03d}.dat")
    
        # Check existence
        if not os.path.exists(dat_filename):
            raise FileNotFoundError(f"Geometry file not found: {dat_filename}")
    
        # Load coordinates
        data = np.loadtxt(dat_filename)
        x, y = data[:, 1], data[:, 2]
    
        # Plot if requested
        if plot_flag:
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, "-")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(f"Airfoil #{airfoil_idx} ({source})")
            plt.xlabel("x")
            plt.ylabel("y",rotation=0,labelpad=10)
            plt.show()
    
        return x, y
    
    def surf_turb(
        self,
        airfoil_number,
        case_number=None,
        Mach=None,
        AoA=None,
        Re=None,
        field_name="Velocity",
        vel_component='a',
        block_index=2,
        action="plot_field",
        **kwargs
    ):
        """
        Locate and process CGNS surface data for a given airfoil and case.
        """
    
        # Step 1: Load CSV
        if not os.path.exists(self.turb_csv):
            print(f"[unifoil] ❌ CSV file not found: {self.turb_csv}")
            return None
    
        df = pd.read_csv(self.turb_csv)
    
        # Determine case number
        if case_number is None:
            if Mach is None or AoA is None or Re is None:
                print("[unifoil] ❌ Must specify either (airfoil, case_number) or (airfoil, Mach, AoA, Re).")
                return None
    
            subset = df[df["airfoil"] == airfoil_number]
            if subset.empty:
                print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
                return None
    
            dist = np.sqrt(
                (subset["Mach"] - Mach) ** 2 +
                (subset["AoA"] - AoA) ** 2 +
                ((subset["Re"] - Re) / subset["Re"].max()) ** 2
            )
            best_idx = dist.idxmin()
            row = subset.loc[best_idx]
            case_number = int(row["case"])
            Mach, AoA, Re = row["Mach"], row["AoA"], row["Re"]
            print(f"[unifoil] Closest match → Airfoil {airfoil_number}, Case {case_number} "
                  f"(Mach={Mach:.3f}, AoA={AoA:.3f}, Re={Re:.2e})")
        else:
            row = df[(df["airfoil"] == airfoil_number) & (df["case"] == case_number)]
            if row.empty:
                print(f"[unifoil] ❌ No entry found for Airfoil {airfoil_number}, Case {case_number}.")
                return None
            Mach, AoA, Re = row["Mach"].values[0], row["AoA"].values[0], row["Re"].values[0]
    
        # Step 2: Find CGNS file
        case_index = case_number - 1  # ✅ 1-indexed → 0-indexed
        filename_pattern = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_000_surf_turb.cgns"
        found_path = None
    
        for i in range(1, 7):
            cutout_folder = os.path.join(self.cwd, f"Turb_Cutout_{i}")
            if not os.path.isdir(cutout_folder):
                continue
            candidate_path = os.path.join(cutout_folder, filename_pattern)
            if os.path.exists(candidate_path):
                found_path = candidate_path
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ CGNS file not found for Airfoil {airfoil_number}, Case {case_number}. "
                  f"(possibly missing or unconverged simulation)")
            return None
    
        print(f"[unifoil] ✅ Found CGNS file: {found_path}")
    
        # Step 3: Geometry file
        airfoil_file = None
        ft_file_candidate = os.path.join(self.ft_geom_path, f"airfoil_{airfoil_number}.dat")
        nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{airfoil_number:03d}.dat")
    
        if os.path.exists(ft_file_candidate):
            airfoil_file = ft_file_candidate
        elif os.path.exists(nlf_file_candidate):
            airfoil_file = nlf_file_candidate
        else:
            print(f"[unifoil] ⚠️ Airfoil geometry file not found for Airfoil {airfoil_number}.")
    
        # Step 4: Execute requested action
        try:
            post = surfCGNSPostProcessor(found_path, airfoil_file=airfoil_file)
            if action == "plot_field":
                post.plot_field(field_name=field_name, block_index=block_index, vel_component=vel_component)
                return None
            elif hasattr(post, action):
                func = getattr(post, action)
                if callable(func):
                    # Always pass field_name and block_index if the function expects them
                    if "field_name" in func.__code__.co_varnames:
                        kwargs.setdefault("field_name", field_name)
                    if "block_index" in func.__code__.co_varnames:
                        kwargs.setdefault("block_index", block_index)
                    if "vel_component" in func.__code__.co_varnames:
                        kwargs.setdefault("vel_component", vel_component)
            
                    result = func(**kwargs)
                    return result

                else:
                    print(f"[unifoil] ⚠️ '{action}' exists but is not callable.")
                    return None
            else:
                print(f"[unifoil] ⚠️ Unknown action '{action}'. Available: "
                      f"{[m for m in dir(post) if not m.startswith('_')]}")
                return None
        except Exception as e:
            print(f"[unifoil] ❌ Error running surfCGNSPostProcessor.{action}: {e}")
            return None
    

    def get_aero_coeffs(self, airfoil_number, case_number, print_flag=True):
        """
        Extract CL and CD from airfoil_<num>_G2_A_L0_case_<num>_analysis.csv.
        Handles both 7-row and 8-row formats, converts text to floats,
        and converts user-provided 1-indexed case_number to 0-indexed filenames.
    
        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as in CSV).
        print_flag : bool, optional
            If True, prints Cl and Cd to console.
    
        Returns
        -------
        tuple : (Cl, Cd) or (None, None) if file not found.
        """
        import pandas as pd
        import numpy as np
        import os
    
        # Convert 1-based case number to 0-based for filename
        case_index = case_number - 1
        filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_analysis.csv"
    
        # Search both datasets
        found_path = None
        for folder in ["airfoil_data_from_simulations_turb_set1", "airfoil_data_from_simulations_turb_set2"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ Analysis file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None, None
    
        try:
            df = pd.read_csv(found_path, header=None)
    
            # Locate CL/CD header
            header_row = None
            for i in range(len(df)):
                vals = [str(v).strip().upper() for v in df.iloc[i, :2].values]
                if "CL" in vals and "CD" in vals:
                    header_row = i
                    break
    
            # Extract row below CL/CD header or fallback to last row
            if header_row is not None and header_row + 1 < len(df):
                Cl_raw, Cd_raw = df.iloc[header_row + 1, 0], df.iloc[header_row + 1, 1]
            else:
                Cl_raw, Cd_raw = df.iloc[-1, 0], df.iloc[-1, 1]
    
            # Convert to float robustly
            try:
                Cl = float(Cl_raw)
                Cd = float(Cd_raw)
            except Exception:
                Cl, Cd = pd.to_numeric([Cl_raw, Cd_raw], errors="coerce")
    
            if np.isnan(Cl) or np.isnan(Cd):
                raise ValueError(f"Invalid numeric values: Cl={Cl_raw}, Cd={Cd_raw}")
    
            if print_flag:
                print(f"[unifoil] ✅ Airfoil {airfoil_number}, Case {case_number} → File Case {case_index}: Cl = {Cl:.6f}, Cd = {Cd:.6f}")
                print(f"[unifoil]    File: {found_path}")
    
            return Cl, Cd
    
        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None, None





    # ==============================================================
    # 2️⃣ Load and optionally print convergence.pkl contents
    # ==============================================================
    def load_convergence_data(self, airfoil_number, case_number, print_flag=False):
        """
        Load convergence pickle data for a given airfoil and case.
        Automatically converts user-provided 1-indexed case_number to 0-indexed filenames.
    
        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as seen in CSV).
        print_flag : bool, optional
            If True, prints a summary of the loaded contents.
    
        Returns
        -------
        dict or None
            Dictionary loaded from pickle, or None if file not found or unreadable.
        """
        import os
        import pickle
        import numpy as np
    
        # Convert 1-based to 0-based case index
        case_index = case_number - 1
        filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_convergence.pkl"
    
        # Search both dataset folders
        found_path = None
        for folder in ["airfoil_data_from_simulations_turb_set1", "airfoil_data_from_simulations_turb_set2"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ Convergence file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None
    
        try:
            with open(found_path, "rb") as f:
                data = pickle.load(f)
    
            if print_flag:
                print(f"[unifoil] ✅ Loaded convergence data for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}):")
                for key, val in data.items():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        print(f"  {key}: array with {len(val)} elements")
                    else:
                        print(f"  {key}: {val}")
    
            return data
    
        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None
