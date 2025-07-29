import os
import sys

# Get absolute path to the UniFoil root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add UniFoil to sys.path
sys.path.append(ROOT_DIR)
from tools.post_processor_class_transi import NFactorPlotter

# Initialize with static paths or use defaults
plotter = NFactorPlotter(
    nfactor_file='./sample_input_data/nfactor_ts.dat',
    airfoil_file='./sample_input_data/airfoil_coords.dat',
    transiloc_file='./sample_input_data/transiLoc.dat'
)

# Choose what to run
plotter.plot_nfactor(save_npz_path='./nfactor_data.npz')
plotter.plot_airfoil_with_transition(save_npz_path='./airfoil_and_transitions.npz')