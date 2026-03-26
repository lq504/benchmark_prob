# benchmark_prob

Code for the benchmark problem.

This folder contains a self-contained notebook demo and sample data files.

## Contents

- `benchmark_prob_demo.ipynb`
- `samples-0000260967.txt`
- `samples-0003505828.txt`
- `samples-0008340947.txt`
- `samples-0009255870.txt`

## How to run

### 1) Create/activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install numpy scipy jax jaxlib matplotlib jupyter
```

### 3) Start Jupyter from this folder

```bash
jupyter notebook
```

Then open `benchmark_prob_demo.ipynb` and run all cells.

## Notes

- Run Jupyter from this directory so relative paths resolve to the local sample `.txt` files.
- If your machine uses a specific JAX backend/environment, launch Jupyter from that same environment.
