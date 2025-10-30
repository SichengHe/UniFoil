import numpy as np
import os

class AirfoilGenerator:
    def __init__(self, airfoil_file, xslice_file, output_folder='airfoil_nlf_geom'):
        self.airfoil_file = airfoil_file
        self.xslice_file = xslice_file
        self.output_folder = output_folder

    def generate(self):
        # Load the data
        airfoil_data = np.loadtxt(self.airfoil_file)  # Load the airfoil shapes
        xslice = np.loadtxt(self.xslice_file)         # Load the x-coordinates

        # Ensure xslice and airfoil_data are compatible
        if airfoil_data.shape[1] != len(xslice):
            print(f"Error: xslice length ({len(xslice)}) does not match airfoil data columns ({airfoil_data.shape[1]}).")
            return

        # Create the folder to store .dat files
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Loop through all airfoils and save each one as a .dat file
        for n in range(airfoil_data.shape[0]):
            # Extract the nth airfoil's y-coordinates
            airfoil_y = airfoil_data[n, :]

            # Define the filename
            dat_filename = os.path.join(self.output_folder, f'airfoil_{n+1:03d}.dat')  # e.g., airfoil_001.dat

            # Prepare x-y coordinate pairs
            coords = list(zip(xslice, airfoil_y))

            # Check if the first and last points are the same
            if not (np.isclose(coords[0][0], coords[-1][0]) and np.isclose(coords[0][1], coords[-1][1])):
                coords.append(coords[0])  # Append the first point to close the airfoil

            # Open the .dat file for writing
            with open(dat_filename, 'w') as file:
                for i, (x, y) in enumerate(coords, start=1):
                    file.write(f"{i:03d} {x:.8e} {y:.8e}\n")

        print(f"All airfoil files have been saved in the '{self.output_folder}' folder.")

if __name__ == "__main__":
    gen = AirfoilGenerator(
        airfoil_file="input_nlf/airfoilys.txt",
        xslice_file="input_nlf/xslice.txt"
    )
    gen.generate()
    print("[unifoil] ✅ Finished generating NLF airfoil geometries.")