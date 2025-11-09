import numpy as np
import os

class AirfoilFTGenerator:
    def __init__(self, basis_file, train_file, valid_file, output_folder="airfoil_ft_geom"):
        self.basis_file = basis_file
        self.train_file = train_file
        self.valid_file = valid_file
        self.output_folder = output_folder

    def generate(self):
        # Load basis and x-coordinates
        basis = np.loadtxt(self.basis_file)
        xslice = basis[0, :]
        modes = basis[1:, :]  # shape (14, npts)

        # Create output folder if not exists
        os.makedirs(self.output_folder, exist_ok=True)

        def process_airfoils(datafile, start_idx):
            data = np.loadtxt(datafile)
            for i, row in enumerate(data):
                coefs = row[:14]  # first 14 are modal coefficients
                yslice = np.dot(coefs, modes)

                # Combine x and y directly, no reordering or closure
                coords = np.column_stack((xslice, yslice))

                # Write to .dat file as-is
                file_index = start_idx + i + 1
                filename = os.path.join(self.output_folder, f"airfoil_{file_index}.dat")
                with open(filename, "w") as f:
                    for j, (x, y) in enumerate(coords):
                        f.write(f"{j+1:03d} {x:.8e} {y:.8e}\n")

            return start_idx + data.shape[0]

        # Process training and validation sets
        idx_after_train = process_airfoils(self.train_file, start_idx=0)
        _ = process_airfoils(self.valid_file, start_idx=idx_after_train)
if __name__ == "__main__":
    # Expect inputs to exist in the *current working directory*
    gen = AirfoilFTGenerator(
        basis_file="input_ft/basis.txt",
        train_file="input_ft/training.dat",
        valid_file="input_ft/validating.dat"
    )
    gen.generate()
    print("[unifoil] ✅ Finished generating FT airfoil geometries.")
