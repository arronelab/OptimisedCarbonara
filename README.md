## Protein Structure Refinement with SAXS and Gradient Descent

This project implements a PyTorch-based model for refining protein structures using a combination of geometric constraints and experimental scattering data. It's designed to optimise molecular conformations by maximizing a likelihood function that incorporates various structural and experimental SAXS factors. Its still very well in-development by Chris Prior (christopher.prior@durham.ac.uk), Arron Bale (arron.n.bale@durham.ac.uk), and Josh Mckeown (josh.j.mckeown@durham.ac.uk). 

## Features

- Incorporate experimental small-angle X-ray scattering (SAXS) data into the refinement process
- Utilise geometric constraints such as bond lengths, angles, and allowed curvature/torsions
- Visualise the predicted structures and compare them with initial conformations
- Track and plot the optimization process to monitor convergence

## Main Components

1. `setUpBackbone.py`: Contains utility functions for setting up the molecular backbone, computing geometric features, and handling SAXS data.
2. `model.ipynb`: Notebook that implements the main `LikelihoodModel` class and training loop.

## Prerequisites

- Python 3.9+
- PyTorch 2.4.1+

see the `requirements.txt` for full list

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/arronelab/OptimisedCarbonara.git
   cd OptimisedCarbonara
   ```

2. Create a new conda environment with the required packages:
   ```
   conda create --name OptimisedCarbonara_env --file requirements.txt

   conda activate OptimisedCarbonara_env
   ```


## License

This project is licensed under the MIT License

## Acknowledgments

- This project uses the [PyTorch](https://pytorch.org/) library for deep learning and optimization.
- The molecular structure visualization is powered by [Plotly](https://plotly.com/).
- [BioPython](https://biopython.org/) is used for handling PDB files and basic molecular operations.



## Contact

For any questions or collaborations, please open an issue in this repository or contact me
