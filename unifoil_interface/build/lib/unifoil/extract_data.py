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
        *,
        field_name="Velocity",
        vel_component='a',
        block_index=2,
        action="plot_field",
        **kwargs
    ):
        """
        Locate and process CGNS surface data for a given airfoil & case.
        Falls back to the nearest AVAILABLE case (with an existing CGNS) if the ideal one is missing.
        """
    
        # ---------- helpers ----------
        def cutout_roots():
            return [os.path.join(self.cwd, f"Turb_Cutout_{i}") for i in range(1, 7)]
    
        def cgns_path_for(airfoil, case_idx0):
            fname = f"airfoil_{airfoil}_G2_A_L0_case_{case_idx0}_000_surf_turb.cgns"
            for root in cutout_roots():
                if os.path.isdir(root):
                    cand = os.path.join(root, fname)
                    if os.path.exists(cand):
                        return cand
            return None
    
        def nearest_available_case(subset, target_M, target_A, target_Re):
            """Return (row, path) for the nearest row with an existing CGNS path."""
            if subset.empty:
                return None, None
    
            # Scale Re for distance metric
            re_scale = max(subset["Re"].max(), 1.0)
            # Compute distances
            d = np.sqrt(
                (subset["Mach"] - target_M) ** 2
                + (subset["AoA"] - target_A) ** 2
                + ((subset["Re"] - target_Re) / re_scale) ** 2
            )
            # Sort by distance, pick first that exists on disk
            order = np.argsort(d.values)
            for idx in order:
                row = subset.iloc[idx]
                path = cgns_path_for(airfoil_number, int(row["case"]) - 1)
                if path is not None:
                    return row, path
            return None, None
    
        # ---------- Step 1: CSV ----------
        if not os.path.exists(self.turb_csv):
            print(f"[unifoil] ❌ CSV file not found: {self.turb_csv}")
            return None
    
        df = pd.read_csv(self.turb_csv)
        subset = df[df["airfoil"] == airfoil_number]
        if subset.empty:
            print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
            return None
    
        # ---------- Step 2: choose target row/case ----------
        target_row = None
        found_path = None
    
        if case_number is not None:
            row = df[(df["airfoil"] == airfoil_number) & (df["case"] == case_number)]
            if row.empty:
                print(f"[unifoil] ❌ No entry for airfoil {airfoil_number}, case {case_number}.")
                return None
            row = row.iloc[0]
            case_idx0 = int(row["case"]) - 1
            path = cgns_path_for(airfoil_number, case_idx0)
    
            if path is not None:
                # exact case exists
                target_row, found_path = row, path
                print(f"[unifoil] ✅ Using requested case → airfoil {airfoil_number}, case {int(row['case'])} "
                      f"(Mach={row['Mach']:.3f}, AoA={row['AoA']:.3f}, Re={row['Re']:.2e})")
            else:
                # fallback to nearest AVAILABLE using this row's conditions (or provided Mach/AoA/Re if given)
                fallback_M = float(Mach) if Mach is not None else float(row["Mach"])
                fallback_A = float(AoA)  if AoA  is not None else float(row["AoA"])
                fallback_R = float(Re)   if Re   is not None else float(row["Re"])
                print(f"[unifoil] ⚠️ CGNS not found for requested case {case_number}. "
                      f"Searching nearest AVAILABLE case for airfoil {airfoil_number}...")
                cand_row, cand_path = nearest_available_case(subset, fallback_M, fallback_A, fallback_R)
                if cand_row is None:
                    print(f"[unifoil] ❌ No AVAILABLE CGNS found for airfoil {airfoil_number}.")
                    return None
                target_row, found_path = cand_row, cand_path
                print(f"[unifoil] 🔁 Fallback → airfoil {airfoil_number}, case {int(cand_row['case'])} "
                      f"(Mach={cand_row['Mach']:.3f}, AoA={cand_row['AoA']:.3f}, Re={cand_row['Re']:.2e})")
        else:
            # No case_number: use nearest AVAILABLE to (Mach, AoA, Re)
            if Mach is None or AoA is None or Re is None:
                print("[unifoil] ❌ Must specify either (airfoil, case_number) OR (airfoil, Mach, AoA, Re).")
                return None
            cand_row, cand_path = nearest_available_case(subset, float(Mach), float(AoA), float(Re))
            if cand_row is None:
                print(f"[unifoil] ❌ No AVAILABLE CGNS found for airfoil {airfoil_number}.")
                return None
            target_row, found_path = cand_row, cand_path
            print(f"[unifoil] Closest AVAILABLE match → airfoil {airfoil_number}, case {int(cand_row['case'])} "
                  f"(Mach={cand_row['Mach']:.3f}, AoA={cand_row['AoA']:.3f}, Re={cand_row['Re']:.2e})")
    
        # ---------- Step 3: geometry (overlay) ----------
        airfoil_file = None
        ft_candidate  = os.path.join(self.ft_geom_path,  f"airfoil_{airfoil_number}.dat")
        nlf_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{airfoil_number:03d}.dat")
        if os.path.exists(ft_candidate):
            airfoil_file = ft_candidate
        elif os.path.exists(nlf_candidate):
            airfoil_file = nlf_candidate
        else:
            print(f"[unifoil] ℹ️  No airfoil geometry file found for airfoil {airfoil_number} (overlay skipped).")
    
        # ---------- Step 4: execute action ----------
        try:
            post = surfCGNSPostProcessor(found_path, airfoil_file=airfoil_file)
    
            if action == "plot_field":
                post.plot_field(field_name, block_index=block_index,
                                vel_component=vel_component, **kwargs)
                return None
    
            if action == "extract_xy_quantity":
                return post.extract_xy_quantity(field_name, vel_component=vel_component,
                                                block_index=block_index, **kwargs)
    
            if action == "available_fields":
                return post.available_fields(block_index=block_index)
    
            if action == "display_structure":
                post.display_structure()
                return None
    
            if hasattr(post, action):
                func = getattr(post, action)
                if callable(func):
                    if "field_name"    in func.__code__.co_varnames: kwargs.setdefault("field_name", field_name)
                    if "block_index"   in func.__code__.co_varnames: kwargs.setdefault("block_index", block_index)
                    if "vel_component" in func.__code__.co_varnames: kwargs.setdefault("vel_component", vel_component)
                    return func(**kwargs)
    
            print(f"[unifoil] ⚠️ Unknown action '{action}'. "
                  f"Try: plot_field, extract_xy_quantity, available_fields, display_structure")
            return None
    
        except Exception as e:
            print(f"[unifoil] ❌ Error in surf_turb('{action}'): {e}")
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
