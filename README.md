**UniFoil** is a Universal Dataset of Airfoils in Transitional and Turbulent Regimes for Subsonic and Transonic Flows for Machine Learning.
It consists of 500,000 simulations covering transitional and fully turbulent flows across incompressible to compressible regimes for 2D airfoils.
UniFoil features include the following:
- 400,000 fully-turbulent (FT) airfoil simulations
- 50,000 natural laminar flow (NLF) airfoil simulations in fully turbulent regime
- 50,000 NLF airfoil simulations in the transition regime.

<p align="center">
  <img src="images/Logo.png" width="200"/>
</p>

The dataset can be found at the Harvard Dataverse here -> https://doi.org/10.7910/DVN/VQGWC4

## How to use UniFoil?
Clone this repository and enter the UniFoil directory.
Run the following command in terminal to install UniFoil interface. **Please ensure that you have a fresh python virtual environment before doing so.**
```html
pip install .
```
This interface will hekp with data visualization and extraction from the dataset. 
For usage, please refer to the ``Read the Docs`` website -> https://unifoildocs.readthedocs.io/
## License

Distributed using the **CC BY-SA** license, version 4.0; \
Please see the LICENSE file for details.

## Citing UniFoil

If you use the UniFoil dataset, please cite

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
```
Also, please cite our paper
```bibtex
@ARTICLE{Kanchi2025,
  title         = "{UniFoil}: A universal dataset of airfoils in transitional
                   and turbulent regimes for subsonic and transonic flows",
  author        = "Kanchi, Rohit Sunil and Melanson, Benjamin and Somasekharan,
                   Nithin and Pan, Shaowu and He, Sicheng",
  month         =  oct,
  year          =  2025,
  copyright     = "http://creativecommons.org/licenses/by-sa/4.0/",
  archivePrefix = "arXiv",
  primaryClass  = "physics.flu-dyn",
  eprint        = "2505.21124"
}
```