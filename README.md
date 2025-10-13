# DiffWake: A General Differentiable Wind-Farm Solver in JAX

**DiffWake** is a fully differentiable implementation of the **Cumulative–Curl (CC)** wake model for wind-farm flow simulation, written in [JAX](https://github.com/google/jax).  

It enables, e.g,  *end-to-end gradient propagation* through wake, thrust, and power computations — supporting modern machine learning (ML) workflows, parameter inference, and gradient-based layout optimization on GPUs.

<p align="center">
  <img src="plots/wake_visualization.png" width="450"/>
</p>

---

## 🚀 Key Features
- **Fully differentiable CC wake model** – exact reverse-mode gradients through all wake, thrust, and power computations.  
- **Physics-consistent formulation** – reproduces the analytic CC equations from [Martínez-Tossas et al. 2019](https://doi.org/10.1016/j.energy.2019.116148) and [Bay et al. 2023](https://doi.org/10.5194/wes-8-401-2023).  
- **Compiled tensor operations** – implemented using `jax.lax.fori_loop` for stable, efficient GPU execution.  
- **Batchable evaluation** – run multiple wind speeds, directions, or parameter sets in parallel.  
- **Gradient-based optimization** – compatible with optimizers such as L-BFGS, Adam, or custom differentiable design loops.  
- **Probabilistic parameter inference** – includes an example for learning turbulence intensity (TI) distributions from SCADA data.  

---

## 📘 Background

Traditional wake models are efficient but not differentiable, limiting their use in modern gradient-based optimization and ML frameworks.  
**DiffWake** bridges that gap by rewriting the Cumulative–Curl (CC) model in pure JAX — allowing:
- Gradient-based layout and control optimization.
- End-to-end parameter calibration directly from observed turbine power.
- Integration with probabilistic or deep-learning models.

For details, see the accompanying paper:

> *DiffWake: A General Differentiable Wind-Farm Solver in JAX*  
> M. Bånkestad, et al. (2025)

---

## 🧭 TODO / Roadmap

Planned features and improvements:

- [ ] **Tilt angles** – not yet supported in the differentiable formulation.  
- [ ] **Jupyter notebooks** – add example notebooks for:  
  - gradient-based layout optimization  
  - turbulence-intensity calibration  
  - comparison with FLORIS baseline  
- [ ] **PyTorch version** – experimental mirror for cross-validation.  
- [ ] **Differentiable yaw control** – extend optimization to $\gamma_\mathrm{yaw}$.  
- [ ] **Parameter calibration** – extend to additional CC parameters ($k_y$, $k_z$, $x_\mathrm{nw}$, $C_T$).  
- [ ] **Improved batching** – enable large wind-rose evaluation without recompilation.  
- [ ] **Documentation and figures** – add user guide and visual examples.

---

## 🧩 Acknowledgments

This project builds upon the Cumulative–Curl (CC) wake formulation and utilizes the reference implementation from the FLORIS framework, developed by the National Renewable Energy Laboratory (NREL). Portions of the numerical algorithms and data structures have been adapted from the original FLORIS codebase, distributed under the BSD-3-Clause License.
 
Original FLORIS Repository: https://github.com/NREL/floris
Original License: BSD-3-Clause License (see licenses/LICENSE_FLORIS.txt)
Copyright: © 2013–2025 Alliance for Sustainable Energy, LLC.
