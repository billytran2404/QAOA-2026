# QAOA-2026
The data and source code for the paper "Solving Multi-Objective Faculty Timetabling Problem via QAOA"

# QAOA for Faculty Timetabling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-latest-green)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A Quantum Approximate Optimization Algorithm (QAOA) implementation for solving the **Faculty Timetabling Problem**, developed as part of ongoing research in quantum computing at Texas Tech University.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

---

## Overview

This repository contains the implementation of QAOA applied to the Faculty Timetabling problem. The qubit count ranges from **16 to 32 qubits** depending on the problem instance size. In this study, experiments are conducted using **20 qubits**.

---

## Repository Structure

```
QAOA-2026/
│
├── dataset/                        # Dataset files for Faculty Timetabling
│   ├── QAOA_toy_16qbit.xlsx        # Problem instance (16 qubits)
│   ├── QAOA_toy_20qbit.xlsx        # Problem instance (20 qubits) ← used in paper
│   ├── QAOA_toy_24qbit.xlsx        # Problem instance (24 qubits)
│   └── QAOA_toy_28qbit.xlsx       # Problem instance (28 qubits)
│   └── QAOA_toy_32qbit.xlsx       # Problem instance (31 qubits)
│
├── algorithm_runner.py          # Main script to run QAOA algorithm
├── visualizer.py                # Generates figures and plots for the paper
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Data

The `dataset/` directory contains dataset files for the **Faculty Timetabling** problem.

| File | Qubits | Description |
|------|--------|-------------|
| `QAOA_toy_16qbit.xlsx` | 16 | Small-scale instance |
| `QAOA_toy_20qbit.xlsx` | 20 | **Used in this study** |
| `QAOA_toy_24qbit.xlsx` | 24 | Medium-scale instance |
| `QAOA_toy_28qbit.xlsx` | 28 | Medium-scale instance |
| `QAOA_toy_32qbit.xlsx` | 32 | Large-scale instance |

> **Note:** The qubit count ranges from **16 to 32**. All experiments reported in the paper are conducted with **20 qubits**.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pennylane` — Quantum circuit simulation
- `numpy` — Numerical computation
- `matplotlib` — Visualization
- `scipy` — Optimization routines

---

## Usage

### Run the QAOA Algorithm

[`algorithm_runner.py`](https://github.com/billytran2404/QAOA-2026/blob/main/algorithm_runner.py) is the main script for executing the QAOA solver.

```bash
python algorithm_runner.py
```

### Generate Figures

`visualizer.py` produces all figures and plots used in the paper.

```bash
python visualizer.py
```

---

## Results

Experiments are conducted on the 20-qubit Faculty Timetabling instance. Results include convergence plots, solution quality metrics, and comparisons against classical baselines — all reproducible via `visualizer.py`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tran2026qaoa,
  title     = {Solving Multi-Objective Faculty Timetabling Problem via QAOA},
  author    = {Tran, Ban Q. and others},
  booktitle = {To be determined},
  year      = {2026},
  publisher = {To be determined}
}
```

---

## Contact

**Ban Q. Tran** — [bantran@ttu.edu](mailto:bantran@ttu.edu)  
Texas Tech University, Lubbock, TX
