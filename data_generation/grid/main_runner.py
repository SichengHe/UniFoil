import os
import glob
import subprocess
from multiprocessing import Pool, Manager

# === SETTINGS ===
dat_folder = "./airfoil_dat_files"
output_folder = "./generated_meshes"
log_folder = "./logs"
os.makedirs(log_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

geometry = "G2"
grid_family = "A"
n_parallel = 128
batch_size = 500
log_file = os.path.join(log_folder, "failed_cases.txt")
timeout_seconds = 30  # ñ Timeout for mesh generation

# === Function to run mesh generation with timeout handling ===
def run_genmesh_wrapper(args):
    dat_file, failed_list = args
    cmd = [
        "python3", "call_genMesh.py",
        "--input", dat_file,
        "--geometry", geometry,
        "--gridFamily", grid_family,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds  # ñ 30-second timeout
        )
        print(f"[OK] {os.path.basename(dat_file)}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {os.path.basename(dat_file)} took longer than {timeout_seconds}s")
        failed_list.append(dat_file)
    except subprocess.CalledProcessError:
        print(f"[FAILED] {os.path.basename(dat_file)}")
        failed_list.append(dat_file)

# === Function to batch and run mesh generation in parallel ===
def batch_and_run():
    all_dats = sorted(glob.glob(os.path.join(dat_folder, "airfoil_*.dat")))
    total = len(all_dats)
    print(f"=Found {total} airfoils in {dat_folder}")

    manager = Manager()
    failed_cases = manager.list()

    for i in range(0, total, batch_size):
        batch = all_dats[i:i+batch_size]
        print(f"\n= Processing airfoils {i+1} to {i+len(batch)}")

        with Pool(n_parallel) as pool:
            pool.map(run_genmesh_wrapper, [(f, failed_cases) for f in batch])

    # Write log for failed cases
    if failed_cases:
        with open(log_file, "w") as f:
            for fail in failed_cases:
                f.write(f"{fail}\n")
        print(f"\nL {len(failed_cases)} airfoils failed. See '{log_file}' for details.")
    else:
        print("\n All airfoils processed successfully!")

# === Entry point ===
if __name__ == "__main__":
    batch_and_run()
