
# Cluster Expansion & Monte Carlo Simulations of InGaN Alloys

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ICET](https://img.shields.io/badge/ICET-Cluster%20Expansion-green)](https://icet.materialsmodeling.org/)

## 📄 Overview
This repository contains the computational workflow and Python scripts developed for the Bachelor Thesis: **"Cluster Expansion Monte-Carlo simulations of InGaN alloys"** (University of Crete, Department of Physics, 2025).

The project investigates the bulk thermodynamics of **$In_{x}Ga_{1-x}N$** alloys using a first-principles-based Cluster Expansion (CE) model combined with canonical Monte Carlo (MC) simulations. The study covers the full compositional range ($x=0.05-0.95$) and temperatures from 0 K to 1800 K.

## 🎯 Key Objectives
* Construction of a **Phase Diagram** ($\Delta\mu - T$) for pseudomorphic InGaN alloys.
* Analysis of thermodynamic stability and mixing energies.
* Identification of **Order-Disorder transitions** via Heat Capacity ($C_v$) peaks.
* Microscopic analysis of atomic ordering using **Warren-Cowley Short-Range Order (SRO)** parameters.

## 💻 Computational Resources
The Monte Carlo simulations and extensive DFT data processing were performed on the **Metropolis HPC Cluster** of the **University of Crete** (Department of Physics).
* **Cluster Specs:** Utilized standard compute nodes for serial MC annealing jobs.
* **Performance:** High-throughput calculations enabled the construction of high-resolution phase diagrams and precise heat capacity convergence.


## 🛠️ Methodology & Tools
The computational approach combines Density Functional Theory (DFT) data with statistical mechanics methods:
* **Cluster Expansion (CE):** Modeled using the **[ICET](https://icet.materialsmodeling.org/)** package.
* **Monte Carlo (MC):** Performed in the Canonical ensemble using the `mchammer` module.
* **Structure Analysis:** Uses **ASE** (Atomic Simulation Environment) for structure manipulation.

### Key Results
* **Stability:** The zero-temperature mixing energy landscape exhibits a single global minimum, indicating suppression of decomposition into pure GaN and InN under biaxial strain.
* **Phase Transitions:** Order-disorder transitions are observed in the 600–750 K range for In concentrations between 20–40%.
* **Ordering:** Pronounced strain-induced ordering motifs (e.g., $\sqrt{3}\times\sqrt{3}$-type) were identified at low temperatures.

## 📂 Repository Structure
The scripts correspond to the methodology described in the thesis appendices:

| Script Name | Description | Thesis Appendix |
| :--- | :--- | :--- |
| `01_dft_to_db.py` | Parses DFT outputs and creates an ASE database (`.db`) with mixing energies. | A.0 |
| `02_optimize_cutoffs.py` | Optimizes cluster cutoffs using LASSO/Bayesian regression and Cross-Validation. | A.1 |
| `03_train_ce.py` | Fits the Cluster Expansion Hamiltonian and validates it against DFT data. | A.2 |
| `04_run_mc.py` | Runs Canonical Monte Carlo simulations to calculate Energy and Heat Capacity ($C_v$). | A.3 |
| `05_phase_diagram.py` | Calculates Chemical Potential diff ($\Delta\mu$) and plots the Phase Diagram. | A.4 |
| `06_thermo_analysis.py` | Computes Entropy and Helmholtz Free Energy of mixing. | A.5 |
| `07_sro_analysis.py` | Calculates Warren-Cowley SRO parameters for in-plane and vertical shells. | A.6 |

## 🚀 Usage

### 1. Prerequisites
Install the required Python packages:
```bash
pip install icet ase numpy pandas matplotlib scipy seaborn