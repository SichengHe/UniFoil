import os
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool

# --------------------------
# CONFIGURABLE SETTINGS
# --------------------------
k = 14  # number of cases per airfoil
lhs_file = "lhs_all_airfoils_flat_transi.csv"
num_workers = 8
output_aero = "output_aero"
output_transi = "output_transi"

# --------------------------
# SIMULATION FUNCTION
# --------------------------
def run_simulation(task):
    airfoil_num, case_num, mach, aoa, reynolds = task
    print(f"Running airfoil {airfoil_num}, case {case_num}...", flush=True)
    try:
        subprocess.run(
            [
                "mpirun", "-np", "16", "python", "run_one_airfoil.py",
                str(airfoil_num), str(mach), str(aoa), str(reynolds), str(case_num)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print(f"[{airfoil_num}, {case_num}]  Success", flush=True)
    except subprocess.CalledProcessError as e:
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/airfoil_{airfoil_num}_case_{case_num}_error.log"
        with open(log_path, "w") as f:
            f.write(f"Airfoil {airfoil_num}, case {case_num} failed with return code {e.returncode}\n")
            f.write(f"Command: {e.cmd}\n")
        print(f"[{airfoil_num}, {case_num}]  Failed (logged)", flush=True)

# --------------------------
# MAIN FUNCTION (mirrors the first script's flow)
# --------------------------
def main(start_airfoil, end_airfoil, flag):
    all_samples = np.loadtxt(lhs_file, delimiter=",", skiprows=1)
    task_dict = {}
    completed_airfoils = set()
    max_completed_airfoil = start_airfoil - 1

    print(f"= Scanning airfoils {start_airfoil} to {end_airfoil}...")

    for airfoil_num in range(start_airfoil, end_airfoil + 1):
        # If flag==1 we'll overwrite task_dict later anyway, but we still scan to print stats consistently
        missing = False

        for case_num in range(k):
            folder_name = f"airfoil_{airfoil_num}_G2_A_L0_case_{case_num}"
            aero_path = os.path.join(output_aero, folder_name)
            transi_path = os.path.join(output_transi, folder_name)

            aero_ok = os.path.isdir(aero_path) and len(os.listdir(aero_path)) == 3
            transi_ok = os.path.isdir(transi_path) and len(os.listdir(transi_path)) == 8

            if not (aero_ok and transi_ok):
                idx = (airfoil_num - 1) * k + case_num
                if idx < len(all_samples):
                    mach, aoa, reynolds = all_samples[idx]
                    task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)
                missing = True

        if not missing:
            completed_airfoils.add(airfoil_num)
            max_completed_airfoil = max(max_completed_airfoil, airfoil_num)

    num_missing = len(task_dict)

    if flag == 1:
        print("» Running all airfoils fresh...")
        task_dict.clear()
        for airfoil_num in range(start_airfoil, end_airfoil + 1):
            for case_num in range(k):
                idx = (airfoil_num - 1) * k + case_num
                if idx < len(all_samples):
                    mach, aoa, reynolds = all_samples[idx]
                    task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)

    elif flag == 2:
        print("» Running missing cases and resuming from last completed airfoil...")
        # keep what's already in task_dict (the missing ones)
        resume_start = max_completed_airfoil + 1
        if resume_start <= end_airfoil:
            print(f"» Resuming from airfoil {resume_start} to {end_airfoil}")
            for airfoil_num in range(resume_start, end_airfoil + 1):
                for case_num in range(k):
                    idx = (airfoil_num - 1) * k + case_num
                    if idx < len(all_samples):
                        mach, aoa, reynolds = all_samples[idx]
                        task_dict[(airfoil_num, case_num)] = (airfoil_num, case_num, mach, aoa, reynolds)
        else:
            print("» All airfoils in range already completed. No resume needed.")

    else:
        print("All airfoils in range already completed. No resume needed.")
        task_dict.clear()

    # Final summary
    total_expected = (end_airfoil - start_airfoil + 1) * k
    num_to_run = len(task_dict)
    num_done = total_expected - num_missing if flag != 1 else 0  # for fresh run, we treat all as to-run

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
        print("No tasks to run.")

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
