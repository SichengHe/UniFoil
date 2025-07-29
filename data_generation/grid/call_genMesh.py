import sys, os
import numpy as np
#from matplotlib import pyplot as plt
from pyhyp import pyHyp
from cgnsutilities import cgnsutilities
from prefoil import Airfoil, sampling

def generate_mesh(input_file, geometry="G2", grid_family="A", debug=False):
    geometry_config = {
        "G1": {"chord": 0.23},
        "G2": {"chord": 1.0},
    }
    chord = geometry_config[geometry]["chord"]

    if grid_family == "A":
        nPts = 293
        N = 85
        nTEPts = 11
        s0 = 5e-7
    elif grid_family == "B":
        nPts = 897
        N = 97
        nTEPts = 15
        s0 = 1e-6

    # Create output directory
    output_dir = "generated_meshes"
    os.makedirs(output_dir, exist_ok=True)

    coords = np.loadtxt(input_file)[:, 1:]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    airfoil_name = f"{base_name}_{geometry}_{grid_family}"

    airfoil = Airfoil(coords, normalize=True)
    airfoil.scale(factor=chord)
    data = airfoil.getSampledPts(
        nPts, spacingFunc=sampling.conical, func_args={"coeff": 1}, nTEPts=nTEPts
    )

    if debug:
        fig1 = airfoil.plot()
        fig1.suptitle(f"{airfoil_name} with selected coordinates sampling")
        #plt.show()

    # Save coordinates in output folder
    plot3DFileName = os.path.join(output_dir, f"{airfoil_name}.xyz")
    os.chdir(output_dir)
    airfoil.writeCoords(airfoil_name)
    os.chdir("..")
    plot3DFileName = os.path.join(output_dir, f"{airfoil_name}.xyz")


    options = {
        "inputFile": plot3DFileName,
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        "N": N,
        "s0": s0 * chord,
        "marchDist": 100 * chord,
    }

    cgnsFileName = os.path.join(output_dir, f"{airfoil_name}_L0.cgns")
    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(cgnsFileName)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .dat file")
    parser.add_argument("--geometry", type=str, default="G2", choices=["G1", "G2"])
    parser.add_argument("--gridFamily", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    generate_mesh(args.input, args.geometry, args.gridFamily, args.debug)
