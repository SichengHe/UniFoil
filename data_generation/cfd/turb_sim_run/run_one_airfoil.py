import numpy as np
import os
import argparse
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI
import csv
import pickle

comm = MPI.COMM_WORLD

# Argument parser: airfoil number, Mach, AoA, Reynolds, and Case number
parser = argparse.ArgumentParser()
parser.add_argument("airfoil_number", type=int, help="Airfoil number (e.g., 123)")
parser.add_argument("mach", type=float, help="Mach number (e.g., 0.7)")
parser.add_argument("aoa", type=float, help="Angle of attack in degrees (e.g., 5.0)")
parser.add_argument("reynolds", type=float, help="Reynolds number (e.g., 5e6)")
parser.add_argument("case_number", type=int, help="Case number identifier")
args = parser.parse_args()

# Inputs
airfoil_num = args.airfoil_number
mach = args.mach
aoa = args.aoa
reynolds = args.reynolds
case_num = args.case_number

airfoil_name = f"airfoil_{airfoil_num}_G2_A_L0"
gridFile = os.path.join("generated_meshes", f"{airfoil_name}.cgns")

# Validate grid file exists
if not os.path.isfile(gridFile):
    raise FileNotFoundError(f"CGNS file not found: {gridFile}")

if comm.rank == 0:
    print(f"Running simulation for {airfoil_name} (Case {case_num})")
    print(f"Mach: {mach}, AoA: {aoa}, Re: {reynolds:.2e}")

# Output directories
#output_subdir = os.path.join("output", airfoil_name)
#data_dir = "airfoil_data_from_simulations"
output_subdir = os.path.join("/anvil/scratch/x-rkanchi/output", airfoil_name)
data_dir = "/anvil/scratch/x-rkanchi/airfoil_data_from_simulations"
if comm.rank == 0:
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

# ADFLOW solver options
aeroOptions = {
    "gridFile": gridFile,
    "outputDirectory": output_subdir,
    "monitorvariables": ["resrho", "cl", "cd"],
    "writevolumesolution": False,
    "writesurfacesolution": True,
    "equationType": "RANS",
    "MGCycle": "sg",
    "useANKSolver": True,
    "useNKSolver": True,
    "NKSwitchTol": 1e-4,
    "L2Convergence": 1e-6,
    "nCycles": 10000,
}

CFDSolver = ADFLOW(options=aeroOptions)

# Define aero problem
ap = AeroProblem(
    name=f"{airfoil_name}_case_{case_num}",
    mach=mach,
    reynolds=reynolds,
    reynoldsLength=1.0,
    T=300,
    alpha=aoa,
    areaRef=1.0,
    chordRef=1.0,
    evalFuncs=["cl", "cd"]
)

# Run simulation
CFDSolver(ap)
funcs = {}
CFDSolver.evalFunctions(ap, funcs)

# Save results
if comm.rank == 0:
    csv_filename = f"{airfoil_name}_case_{case_num}_analysis.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Airfoil", airfoil_name])
        writer.writerow(["Case Number", case_num])
        writer.writerow(["Mach", mach])
        writer.writerow(["AoA", aoa])
        writer.writerow(["Reynolds", f"{reynolds:.2e}"])
        writer.writerow([])
        writer.writerow(["CL", "CD"])
        writer.writerow([funcs[f"{ap.name}_cl"], funcs[f"{ap.name}_cd"]])
    print(f" Saved: {csv_path}")

# Save convergence history
hist = CFDSolver.getConvergenceHistory()
if comm.rank == 0:
    convergence_file = os.path.join(data_dir, f"{airfoil_name}_case_{case_num}_convergence.pkl")
    with open(convergence_file, "wb") as f:
        pickle.dump(hist, f)
    print(f"Convergence history saved: {convergence_file}")

