## Technical details

In the following, we will go over more technical details of the model. However, for a **mathematical** description
of the model, please check the manuscript on [arXiv](https://doi.org/10.48550/arXiv.2605.24544).

### Numerical and physical features

**Physics**

- **Prognostic equations** for horizontal wind components (u, v), potential temperature ($\Theta$),
  specific humidity ($Q_v$), and turbulent kinetic energy (TKE).
- **Turbulence closure**: Mellor-Yamada-Nakanishi-Niino Level-2.5 (MYNN 2.5) following
  [Nakanishi & Niino (2009)](https://doi.org/10.2151/jmsj.87.895).
- Surface coupling via **Monin-Obukhov Similarity Theory (MOST)** allowing to prescribe
    - surface temperature or sensible heat flux
    - surface moisture flux
- Large scale forcing via time-dependent geostrophic wind profiles and latitude-dependent Coriolis force

**Time integration**

- **Explicit**: 2nd-order Adams-Bashforth (AB2) with optional CFL-based adaptive $\Delta t$.
- **Semi-Implicit**: Crank-Nicolson (CN) for vertical diffusion, combined with AB2 for explicit source terms.

**Spatial discretization**

Staggered vertical grid with

- **Full levels** (cell centers, `z = dz * (0.5, 1.5, ...)`): prognostic state variables
- **Half levels** (cell faces, `zh = dz * (0, 1, ...)`): turbulent fluxes and diffusivities

where half-level gradients are second-order accurate by using central differences between full levels.

- Time-dependent geostrophic wind profiles via callable
- Coriolis forcing
- ERA5 interfaces for realistic large-scale forcing and initial conditions (untested)

## Setup your own simulation

### The `Simulation` object

todo

### The `Namelist` object/file

todo
