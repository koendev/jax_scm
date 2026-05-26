# JAX-SCM: 1D atmospheric boundary layer meteorology in JAX

![coverage](docs/coverage-badge.svg)
![tests](docs/test-badge.svg)

JAX-SCM is a modern single-column model (SCM) for atmospheric boundary layer simulation, implemented
in [JAX](https://github.com/jax-ml/jax). It implements the **MYNN 2.5 turbulence closure** with **Monin-Obukhov surface
layer** coupling, and is designed for research use.

> [!TIP]
> To get a quick taste of JAX-SCM, you can **run the GABLS1 stable boundary layer case in your browser**!
> Just click the "Open in Colab" button below, which will open an [example notebook](examples/GABLS1_interactive.ipynb)
> in [Google Colab](https://colab.research.google.com/).
>
> <a target="_blank" href="https://colab.research.google.com/github/mpierzyna/jax_scm/blob/main/examples/GABLS1_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
>
> No local installation is required!

**Quick links**:

- [Model description paper (arXiv preprint)](https://doi.org/10.48550/arXiv.2605.24544)
- [API documentation and technical details](https://mpierzyna.github.io/jax_scm/)

## Convective boundary layer example

The plots below show example output of JAX-SCM simulating a moist convective boundary layer.
Initial conditions come from observations, more specifically, from soundings of mean wind, temperature, and humidity.
Realistic sine-like time-varying surface heat and moisture fluxes serve as lower boundary conditions.
The case run here is day 33 of the 1967 Wangara field campaign. Check the
preprint [[arXiv]](https://doi.org/10.48550/arXiv.2605.24544) for more details.

![Wangara Day 33 simulation](docs/res_WG33.png)

## Model description

For a detailed description of the model, please see the preprint
on [[arXiv]](https://doi.org/10.48550/arXiv.2605.24544). If you find JAX-SCM useful for your research,
please cite it as

> Pierzyna, Maximilian. “JAX-SCM v1.0: A Modern Atmospheric Single-Column Model for Boundary Layer Research.”
> arXiv:2605.24544, arXiv, May 2026. https://doi.org/10.48550/arXiv.2605.24544.

or

```
@misc{pierzyna2026a,
  title = {{{JAX-SCM}} v1.0: A Modern Atmospheric Single-Column Model for Boundary Layer Research},
  author = {Pierzyna, Maximilian},
  year = 2026,
  number = {arXiv:2605.24544},
  eprint = {2605.24544},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2605.24544},
}
```

## Quickstart

### Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for package management. Install it first if you don't have it.
Then clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd jax-scm
uv sync  # CPU only
```

To install JAX with CUDA GPU support, run `uv sync --extra cuda` instead.

Verify the installation:

```bash
uv run python -c "import scm; print('OK')"
```

### Simulation setup

Run your own simulations from the `workspaces/` directory. Create a new file, e.g., `my_run.py` in a subdirectory and
set up the simulation as follows.

All simulation parameters (initial conditions, forcing, time stepping, ...) are controlled by the `Simulation`
object. See examples in [`scm.examples`](src/scm/examples). Once the `Simulation` is built, initialize a `Model` and
run the time-stepping loop with `simulate()`. The output is a dictionary of arrays, which can be converted to an
`xarray.Dataset` for analysis and export.

```python
from scm.config import Namelist, TimeIntMethod
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

sim = get_gabls1(Nz=64)  # build Simulation object
cfg = Namelist(time_int=TimeIntMethod.IMPLICIT)  # choose time integration
model = init_model(sim, cfg=cfg)
out = simulate(model=model, sim=sim, cfg=cfg)

ds = out_to_ds(out=out, sim=sim)  # convert to xarray Dataset
ds.to_netcdf("out.nc")
```

Run the simulation as

```bash
uv run python my_run.py
```

Alternatively, you can run this simulation in your browser using Google Colab:
<a target="_blank" href="https://colab.research.google.com/github/mpierzyna/jax_scm/blob/main/examples/GABLS1_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Validation cases

JAX-SCM comes with pre-configured benchmark cases documented in `scm.examples`. Reports verifying the correct
implementation of JAX-SCM against published results for theses cases are contained in the `validation` directory
or are accessible from the table below.

Each case is available for both the Adams-Bashforth 2 (AB2) and Crank-Nicolson (CN) time integration schemes.

| Case                        | AB2                                                        | CN                                                        |
|-----------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| GABLS1 (Cuxart et al. 2006) | [report](validation/gabls1/report_gabls1_ab2.html)         | [report](validation/gabls1/report_gabls1_cn.html)         |
| Andren et al. 1994          | [report](validation/andren1994/report_andren1994_ab2.html) | [report](validation/andren1994/report_andren1994_cn.html) |
| Wangara day 33              | [report](validation/wangara/report_wangara_ab2.html)       | [report](validation/wangara/report_wangara_cn.html)       |
