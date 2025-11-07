# NV Cluster Classifier — simulation with a single NV center

Python simulation of a **one-qubit cluster classifier** using an **NV center in diamond**.  
It builds quantum states via Rx/Ry pulses (with physical NV parameters), evaluates **fidelity vs. label states**, optimizes a simple **cost function** with gradient steps, and visualizes results (cost curve + Bloch sphere).

## What this shows (for recruiters)

- Solid **Python** for scientific computing (NumPy, SciPy, Matplotlib).  
- Modeling with **physical NV parameters** (zero-field splitting, gyromagnetic ratio, detuning).  
- A small but complete loop: **data generation → quantum ops → cost → gradient update → plots**.  
- Clear visualization: cost over iterations and **Bloch sphere** scatter.

## Files

- `nv_cluster_classifier.py` — main script (simulation + plots).  
  *(Current code originates from `NV_machine_learning.py`.)*

## Requirements

```
python 3.9+
numpy
scipy
matplotlib
```

Install:

```bash
pip install numpy scipy matplotlib
```

## Quick start

Run the simulation as is:

```bash
python nv_cluster_classifier.py
```

What you’ll see:
- **Rotation tests** (sanity check): Rx/Ry projections vs. angle.  
- **Synthetic dataset** of cluster points on the (θ, φ) plane → mapped to quantum states.  
- **Optimization loop** (finite-difference gradient) minimizing a cost that mixes:
  - cluster compactness in coordinate space, and
  - prediction confidence via **fidelity** to label states.  
- **Plots**:
  - cost value vs. iteration,  
  - Bloch sphere scatter for labels and final states (color by predicted label).

## How it works (high level)

- **NV model:**  
  Unitaries `Rx(α)` / `Ry(α)` built from a Hamiltonian with  
  - `D = 2870 MHz` (zero-field splitting),  
  - `γ_e = 28024 MHz/T` (electron γ),  
  - `B` (Tesla), `Ω` (MHz), `detuning` (MHz).  
  Evolution via `U = exp(i 2π H t)`.

- **Labels:**  
  A small set of **density matrices** on the Bloch sphere (`StateLabels()`), created from points generated with a Fibonacci sphere.

- **Prediction:**  
  For each quantum state ρ, compute **fidelity** with each label; take `argmax`.

- **Cost (`CostFunction`):**  
  Encourages that samples predicted in the same class are **close** in input space and **confident** (high fidelities).

- **Optimization:**  
  Finite-difference gradient on a 2-param set (`params` for Rx/Ry) with step `st` and learning rate `lr`.

## Parameters you may tweak (inside the script)

- Dataset: `number_clusters`, `points_per_cluster`, `width`, `centers`  
- Physics: `B`, `Omega`, `detuning`  
- Optimization: `number_iterations`, `st`, `lr`, `_lambda`

## Roadmap (short)

- Split plotting helpers into `plots.py` (cleaner main file).  
- Add CLI flags (e.g., `--iters`, `--lr`, `--lambda`, `--clusters`).  
- Save outputs (`results/`) and a reproducible seed/config snapshot.  
- Add a simple **unit test** for `Fidelity` and `BlochCoordinates`.
