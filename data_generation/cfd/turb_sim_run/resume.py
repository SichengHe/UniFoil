import os
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool

# --------------------------
# CONFIGURABLE SETTINGS
# --------------------------
k = 14  # number of cases per airfoil
lhs_file = "lhs_all_airfoils_flat_lam.csv" # For NLF. Fo FT, replace lam with turb in filename.
num_workers = 128
output_dir = "output"

# --------------------------
# SIMULATION FUNCTION
# --------------------------
def run_simulation(task):
    airfoil_num, case_num, mach, aoa, reynolds = task
    print(f"Running airfoil {airfoil_num}, case {case_num}...", flush=True)
    try:
        subprocess.run([
            "python", "run_one_airfoil.py",
            str(airfoil_num), str(mach), str(aoa), str(reynolds), str(case_num)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[{airfoil_num}, {case_num}]  Success", flush=True)
    except subprocess.CalledProcessError as e:
        log_path = f"logs/airfoil_{airfoil_num}_case_{case_num}_error.log"
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Airfoil {airfoil_num}, case {case_num} failed with return code {e.returncode}\n")
            f.write(f"Command: {e.cmd}\n")
        print(f"[{airfoil_num}, {case_num}]  Failed (logged)", flush=True)

# --------------------------
# MAIN FUNCTION
# --------------------------
def main(start_airfoil, end_airfoil, flag):
    all_samples = np.loadtxt(lhs_file, delimiter=",", skiprows=1)
    task_dict = {}
    completed_airfoils = set()
    max_completed_airfoil = start_airfoil - 1

    print(f"= Scanning airfoils {start_airfoil} to {end_airfoil}...")

    for airfoil_num in range(start_airfoil, end_airfoil + 1):
        folder = f"airfoil_{airfoil_num}_G2_A_L0"
        folder_path = os.path.join(output_dir, folder)

        if not os.path.isdir(folder_path) or not os.listdir(folder_path):
            for case_num in range(k):
                idx = (airfoil_num - 1) * k + case_num
                if idx < len(all_samples):
                    mach, aoa, reynolds = all_samples[idx]
                    task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)
            continue

        missing = False
        for case_num in range(k):
            expected_file = f"{folder}_case_{case_num}_000_surf.cgns"
            if expected_file not in os.listdir(folder_path):
                idx = (airfoil_num - 1) * k + (case_num)
                if idx < len(all_samples):
                    mach, aoa, reynolds = all_samples[idx]
                    task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)
                missing = True

        if not missing:
            completed_airfoils.add(airfoil_num)
            max_completed_airfoil = max(max_completed_airfoil, airfoil_num)

    num_missing = len(task_dict)

    if flag == 1:
        print("Â Running all airfoils fresh...")
        task_dict.clear()
        for airfoil_num in range(start_airfoil, end_airfoil + 1):
            for case_num in range(k):
                idx = (airfoil_num - 1) * k + case_num
                if idx < len(all_samples):
                    mach, aoa, reynolds = all_samples[idx]
                    task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)

    elif flag == 2:
        print("Â Running missing cases and resuming from last completed airfoil...")
    
    else:
        print(" All airfoils in range already completed. No resume needed.")
        task_dict.clear()
    

    # Final summary
    total_expected = (end_airfoil - start_airfoil + 1) * k
    num_to_run = len(task_dict)
    num_done = total_expected - num_missing

    print("=" * 50)
    print(f"= Total expected simulations : {total_expected}")
    print(f"= Missing simulations        : {num_missing}")
    print(f"= Completed simulations      : {num_done}")
    print(f"= Simulations to run now     : {num_to_run}")
    print("=" * 50)

    if num_to_run > 0:
        with Pool(processes=num_workers) as pool:
            pool.map(run_simulation, list(task_dict.values()))
    else:
        print(" No tasks to run.")

# --------------------------
# ARGPARSE
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_airfoil", type=int, required=True)
    parser.add_argument("--end_airfoil", type=int, required=True)
    parser.add_argument("--flag", type=int, choices=[1, 2], default=1,
                        help="1 = full run; 2 = restart + resume")
    args = parser.parse_args()

    main(args.start_airfoil, args.end_airfoil, args.flag)
