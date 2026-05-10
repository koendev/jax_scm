import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

from scm.grid import StaggeredGrid
from scm.interfaces import Forcing, Simulation
from scm.mo import BusingerDyerAltSimFuncs, MOSettings
from scm.mynn.interfaces import ProgVarsMYNN


def _const(value, t_s):
    return value


def _linear_in_time(a, b, t_s):
    return a + b * t_s


def _t_to_hours(t_s):
    return t_s / 3600


def get_gabls1(Nz: int = 64, plot: bool = False) -> Simulation:
    """Get a GABLS1 simulation setup.

    References
    ----------
    Cuxart, J., et al. “Single-Column Model Intercomparison for a Stably Stratified Atmospheric Boundary Layer.”
    Boundary-Layer Meteorology, vol. 118, no. 2, Feb. 2006, pp. 273–303.
    https://doi.org/10.1007/s10546-005-3780-1.

    """
    ## Grid
    grid = StaggeredGrid(H=400, Nz=Nz)
    z_inv = 100

    ## Forcing
    # Geostrophic wind
    ug = jnp.ones(Nz) * 8.0  # m/s
    vg = jnp.zeros(Nz)  # m/s

    # Surface temperature forcing: 0.25 K per hour cooling.
    # Use jax.tree_util.Partial so captured arrays remain pytree leaves and
    # this Forcing can be stacked across ensemble members (see scm.ensemble).
    th_s_0 = jnp.array(265.0)  # K
    th_s_rate = jnp.array(-0.25 / 3600.0)  # K/s
    th_s_fn = jax.tree_util.Partial(_linear_in_time, th_s_0, th_s_rate)

    # No moisture
    w_qv_s = jax.tree_util.Partial(_const, jnp.array(0.0))  # kg/kg m/s

    forcing = Forcing(
        u_geo=jax.tree_util.Partial(_const, ug),
        v_geo=jax.tree_util.Partial(_const, vg),
        f_c=1.39e-4,  # 1/s, ~73 deg latitude
        th_s=th_s_fn,
        w_qv_s=w_qv_s,
    )

    # MO settings
    mo_settings = MOSettings(
        z0m=0.1,
        z0h=0.1,
        sim_funcs=BusingerDyerAltSimFuncs(gamma_m=16, gamma_h=16, b_m=4.8, b_h=7.8),
    )

    ## Initial conditions
    # Initial wind profile
    u = jnp.copy(ug)
    v = jnp.copy(vg)

    # Initial temperature
    th = jnp.ones(Nz) * 265.0  # K
    th = jnp.where(grid.z > z_inv, th + 0.01 * (grid.z - z_inv), th)  # capping inversion

    # No moisture
    qv = jnp.zeros(grid.Nz)

    # Initial TKE
    tke = jnp.zeros(grid.Nz)
    tke = jnp.where(grid.z < 250, 0.4 * (1 - grid.z / 250) ** 3, tke)  # m^2 s^-2

    init = ProgVarsMYNN(u=u, v=v, th=th, qke=2 * tke, qv=qv)

    if plot:
        # Initial conditions
        fig, axarr = plt.subplots(ncols=3, figsize=(8, 2), sharey="row", layout="constrained")
        axarr[0].plot(u, grid.z, label="u")
        axarr[0].plot(v, grid.z, label="v")
        axarr[0].set_xlabel("Wind (m/s)")
        axarr[0].set_ylabel("Height (m)")
        axarr[0].legend()

        axarr[1].plot(th, grid.z)
        axarr[1].set_xlabel("Potential Temperature (K)")

        axarr[2].plot(tke, grid.z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

        # Forcing
        fig, axarr = plt.subplots(ncols=2, figsize=(8, 2), width_ratios=[1, 3], layout="constrained")
        axarr[0].plot(ug, grid.z, label="ug")
        axarr[0].plot(vg, grid.z, label="vg")
        axarr[0].set_xlabel("Geostrophic Wind (m/s)")
        axarr[0].legend()

        t = jnp.array([0, 9 * 60 * 60])  # 0 and 9 hours
        axarr[1].plot(t, th_s_fn(t))
        axarr[1].set_xlabel("Time, s")
        axarr[1].set_ylabel("Surface Potential Temperature (K)")

        fig.show()

    return Simulation(
        name="GABLS1",
        grid=grid,
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        t_start_s=0,
        t_end_s=9 * 60 * 60,
        th_ref=263.5,  # midpoint of surface cooling range, following microhh GABLS1 case
        t_index_fn=jax.tree_util.Partial(_t_to_hours),  # hours
    )


def postproc_gabls1(ds: xr.Dataset) -> xr.Dataset:
    """Postprocess GABLS1 output to compute additional diagnostics."""
    m = (ds["u"] ** 2 + ds["v"] ** 2) ** 0.5  # wind magnitude
    tau = (ds["u_w"] ** 2 + ds["v_w"] ** 2) ** 0.5  # total stress
    blh = (tau / tau.isel(zh=0)).where(lambda x: x < 0.05).idxmax("zh")  # blh where stress < 5% of surface stress
    blh /= 0.95  # linear extrapolation (Cuxart et al 2006; Beare et al 2006)

    return xr.Dataset({"m": m, "tau": tau, "blh": blh})
