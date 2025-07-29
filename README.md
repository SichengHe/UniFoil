**UniFoil** is a Universal Dataset of Airfoils in Transitional and Turbulent Regimes for Subsonic and Transonic Flows for Machine Learning.
It consists of 500,000 simulations covering transitional and fully turbulent flows across incompressible to compressible regimes for 2D airfoils.
UniFoil features include the following:
- 400,000 fully-turbulent (FT) airfoil simulations
- 50,000 natural laminar flow (NLF) airfoil simulations in fully turbulent regime
- 50,000 NLF airfoil simulations in the transition regime.

<p align="center">
  <img src="examples/images/Logo.png" width="200"/>
</p>

The dataset can be found here - https://doi.org/10.7910/DVN/VQGWC4

## How to use UniFoil?
This repository provides a modular pipeline for:
- Geometry sampling
- Mesh/grid generation
- CFD simulation
- Dataset extraction & visualization
- Machine learning workflow

---

**Repository Structure**

```
UniFoil/
├── data_generation/
│   ├── geometry/        # Geometry sampling and perturbation scripts
│   ├── grid/            # Grid/mesh generation utilities (e.g., PyHyp, Prefoil)
│   └── cfd/             # CFD simulation drivers and setups
│
├── tools/               # Utilities for dataset processing & ML
│   ├── extractors/      # Data extractors (e.g., CGNS -> npz, vtk)
│   ├── visualizers/     # Post-processing and visualization tools
│   └── ml_helpers/      # Support functions for ML preprocessing
│
├── ml/                  # ML workflows: training, evaluation, models
│   ├── models/          # Neural network architectures (e.g., CNNs, GNNs)
│   ├── training/        # Training loops, hyperparameter config
│   └── evaluation/      # Post-training performance analysis
│
├── examples/            # End-to-end usage examples
│   ├── dataset_post/    # Dataset inspection and plotting
│   └── ml_usage/        # ML inference or benchmark examples
│
├── input/               # Shared input files (optional airfoils, configs, etc.)
├── output/              # Output directory for plots, npz files, predictions
└── README.md
```

---

## Getting Started

1. **Generate data:**
   - Use scripts inside `data_generation/geometry`, `grid`, and `cfd` to produce CFD cases.

2. **Extract features:**
   - Use `tools/extractors/` to convert CGNS data into `.npz` or `.csv`.

3. **Visualize flow:**
   - Use `tools/visualizers/` or `examples/dataset_post/` to understand how to extract data and generate plots or field overlays.

4. **Train ML models:**
   - Run workflows in the `ml/` directory using processed `.npz` or `.csv` datasets.

5. **Test/Deploy:**
   - Use `examples/ml_usage/` to test or deploy trained models.

Each folder contains its own **README.md** file to aid better understanding of information and code.

<p align="center">
  <img src="examples/images/Flow_Regimes.png" width="800"/>
</p>

## Dependencies
The following dependencies are necessary for smooth working of the codes in this repository:
- Tensorflow
- Scikit-learn
- Numpy
- Matplotlib
- PyVista
- Nvidia Cuda Drivers (Optional, for speed)
- ADflow
- ADflow with transition simulation capability (code not released, please contact author)
- pyHyp
- niceplots

### Install all using pip:

```html
pip install tensorflow scikit-learn numpy matplotlib pyvista
```
## License

Distributed using the **CC BY-SA** license, version 4.0; \
Please see the LICENSE file for details.

## Citing UniFoil

If you use UniFoil, please cite:

```bibtex
@misc{UniFoil,
  doi = {10.7910/DVN/VQGWC4},
  url = {https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/VQGWC4},
  author = {Kanchi, Rohit; Melanson, Benjamin; Somasekharan, Nithin; Pan, Shaowu; He, Sicheng},
  keywords = {Engineering, Computer and Information Science, Physics},
  title = {UniFoil},
  publisher = {Harvard Dataverse},
  year = {2025}
}
