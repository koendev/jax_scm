from __future__ import annotations

import dataclasses
import logging
from typing import Tuple, List, Literal, TypeVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import scm
from scm.closures.ysu import init_ysu_closure
from scm.interfaces import DiagVars, ProgVars, StaticForcing, ModelFn, ClosureFn, TransientForcing

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("scm")

T = TypeVar("T")


@dataclasses.dataclass
class SurfaceProperties:
    """Surface properties for the model."""

    z0m: float
    z0h: float
    sim_funcs: scm.mo.MOSimilarityFuncs
    prescribe: Literal["th_s", "w_th_s"]  # todo: I don't like this here

    @property
    def mh_ratio(self):
        return self.z0m / self.z0h


def init_model(grid: scm.grid.StaggeredGrid, sfc: SurfaceProperties, closure_fn: ClosureFn) -> ModelFn:

    # Create MO model
    z_mo = float(grid.z[0])
    eval_mo = scm.mo.init_mo_sfc(
        z0m=sfc.z0m,
        z0h=sfc.z0h,
        z=z_mo,
        z_grad=z_mo / 2,  # Halfway between surface and first full level
        sim_funcs=sfc.sim_funcs,
        prescribe=sfc.prescribe,
    )

    @jax.jit
    def _model(state: ProgVars, forcing: StaticForcing) -> Tuple[ProgVars, DiagVars]:
        # Unpack state
        u, v, th, q = state.u, state.v, state.th, state.q

        # Unpack forcing
        f_c = forcing.f_c
        u_geo, v_geo = forcing.u_geo, forcing.v_geo
        w_th_s, th_s, w_q_s = forcing.w_th_s, forcing.th_s, forcing.w_q_s

        # Run MO for surface coupling
        mo_res: scm.mo.MOResult = eval_mo(u_0=u[0], v_0=v[0], th_0=th[0], w_th_s=w_th_s, th_s=th_s, w_q_s=w_q_s)

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = jnp.zeros(grid.Nz_h)
        du_dz = du_dz.at[1:-1].set((u[1:] - u[:-1]) / grid.dz)
        du_dz = du_dz.at[0].set(mo_res.du_dz)

        dv_dz = jnp.zeros(grid.Nz_h)
        dv_dz = dv_dz.at[1:-1].set((v[1:] - v[:-1]) / grid.dz)
        dv_dz = dv_dz.at[0].set(mo_res.dv_dz)

        dth_dz = jnp.zeros(grid.Nz_h)
        dth_dz = dth_dz.at[1:-1].set((th[1:] - th[:-1]) / grid.dz)
        dth_dz = dth_dz.at[0].set(mo_res.dth_dz)
        dth_dz = dth_dz.at[-1].set(forcing.dth_dz_top)

        dq_dz = jnp.zeros(grid.Nz_h)
        dq_dz = dq_dz.at[1:-1].set((q[1:] - q[:-1]) / grid.dz)
        dq_dz = dq_dz.at[0].set(mo_res.dq_dz)

        # PBL SCHEME on half levels
        grads = ProgVars(u=du_dz, v=dv_dz, th=dth_dz, q=dq_dz)
        diag = closure_fn(state, grads, mo_res)
        u_w, v_w, w_th, w_q = diag.u_w, diag.v_w, diag.w_th, diag.w_q  # unpack

        # Update fluxes with MO results
        # todo: Not needed for YSU, I think because fluxes are already taken care of in PBL scheme
        # u_w = u_w.at[0].set(mo_res.u_w)
        # v_w = v_w.at[0].set(mo_res.v_w)
        # w_th = w_th.at[0].set(mo_res.w_th)

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_th = (w_th[1:] - w_th[:-1]) / grid.dz
        div_w_q = (w_q[1:] - w_q[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f_c * v - f_c * v_geo - div_u_w
        v_tend = -f_c * u + f_c * u_geo - div_v_w
        th_tend = -div_w_th
        q_tend = -div_w_q

        # Gather tendencies and updated diagnostics
        tends = ProgVars(u=u_tend, v=v_tend, th=th_tend, q=q_tend)
        diag = update_dc_obj(diag, u_w=u_w, v_w=v_w, w_th=w_th)
        return tends, diag

    return _model


def update_dc_obj(d: T, **updates) -> T:
    """Update dataclass object with new values."""
    d_dict = dataclasses.asdict(d)
    d_dict.update(updates)
    return d.__class__(**d_dict)


def simulate(
    model: ModelFn,
    ic: ProgVars,
    forcing: TransientForcing,
    dt_s: float,
    t_end_s: float,
    dt_out_s: float,
    ode_int: scm.odeint.METHODS,
) -> Tuple[ProgVars, DiagVars, jnp.ndarray]:
    # Setup time arrays
    t_outer = jnp.arange(0, t_end_s, dt_out_s)
    rel_t_inner = jnp.arange(0, dt_out_s, dt_s)
    jax.debug.print(
        f"Inner steps: {len(rel_t_inner)}, "
        f"Outer steps: {len(t_outer)}, "
        f"Total steps: {len(t_outer) * len(rel_t_inner)}"
    )

    # Create time stepper
    model_stepper = scm.odeint.init_time_stepper(model, dt=dt_s, method=ode_int)

    # Create forcing evaluation function
    get_forcing = forcing.get_eval_fn()

    @jax.jit
    def _scan_inner(carry, t):
        """Advance model by one step but don't accumulate outputs"""
        (state, _) = carry
        state_next, diag_next = model_stepper(state, forcing=get_forcing(t))
        # jax.debug.print("{t}", t=t)
        return (state_next, diag_next), None

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate outputs"""
        (state, _) = carry
        (state_next, diag_next), _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
        jax.debug.print("t={t} ({frac_done:.2f}%)", t=t + dt_out_s, frac_done=100 * (t + dt_out_s) / t_end_s)
        return (state_next, diag_next), (state_next, diag_next)

    # Perform one step to get init DiagVars object, which we can use to initialize the scan
    _, diag_init = model(ic, forcing=get_forcing(jnp.array(0.0)))

    jax.debug.print("Begin simulation...")
    _, (state_hist, diag_hist) = jax.lax.scan(_scan_outer, init=(ic, diag_init), xs=t_outer)
    return state_hist, diag_hist, t_outer


def plot_state(state: scm.interfaces.ProgVars, grid: scm.grid.StaggeredGrid):
    """Plot initial conditions."""
    fig, (ax_uv, ax_th, ax_q) = plt.subplots(ncols=3, figsize=(8, 3), constrained_layout=True)
    ax_uv.plot(state.u, grid.z)
    ax_uv.plot(state.v, grid.z)
    ax_th.plot(state.th, grid.z)
    ax_q.plot(state.q, grid.z)
    fig.show()


def plot_hist(hist: List, t: jnp.ndarray, grid: scm.grid.StaggeredGrid, plot_sfc_val: bool, cmap: str = "viridis"):
    """Plot history of diagnostics."""

    def _plot_profiles(keys):
        fig, axarr = plt.subplots(ncols=len(keys), figsize=(len(keys) * 1.5, 3), sharey="all", constrained_layout=True)
        colors = plt.get_cmap(cmap)(jnp.linspace(0, 1, len(hist)))
        for item, c in zip(hist, colors):
            for ax, k in zip(axarr, keys):
                vals = getattr(item, k)
                z = grid.z if len(vals) == grid.Nz else grid.zh
                if not plot_sfc_val:
                    vals = vals[1:]
                    z = z[1:]
                ax.plot(vals, z, color=c)

        for ax, k in zip(axarr, keys):
            ax.set_xlabel(k)
        axarr[0].set_ylabel("Height (m)")

        fig.show()

    hist_dict = dataclasses.asdict(hist[0])
    keys_profile = []
    keys_ts = []
    for k, v in hist_dict.items():
        if v.ndim == 1:
            keys_profile.append(k)
        elif v.ndim == 0:
            keys_ts.append(k)
        else:
            raise ValueError(f"Unexpected dimension for key '{k}': {v.ndim}")

    if keys_profile:
        _plot_profiles(keys_profile)
    if keys_ts:
        plot_sfc_hist(hist, t=t, keys=keys_ts)


def plot_sfc_hist(hist: List[DiagVars | ProgVars], t: jnp.ndarray, keys: List[str] = None):
    """Plot history of diagnostics at surface."""
    if keys is None:
        keys = list(dataclasses.asdict(hist[0]).keys())

    sfc_vals = {k: [] for k in keys}
    for item in hist:
        for k in keys:
            v = getattr(item, k)
            if v.ndim == 0:  # Single value
                sfc_vals[k].append(v)
            elif v.ndim == 1:  # Profile
                sfc_vals[k].append(v[0])  # Take the first value (surface value)
            else:
                raise ValueError(f"Unexpected dimension for key '{k}': {v.ndim}")

    fig, axarr = plt.subplots(nrows=len(keys), figsize=(5, len(keys) * 1), sharex="all", constrained_layout=True)
    for ax, k in zip(axarr, keys):
        ax.plot(t, sfc_vals[k])
        ax.set_xlabel("Time, s")
        ax.set_ylabel(k)

    fig.show()


def unstack_hist(v: T) -> List[T]:
    v_dict = dataclasses.asdict(v)
    v_class = v.__class__
    n, _ = v_dict[next(iter(v_dict))].shape  # Get number of time steps
    return [v_class(**{k: v_dict[k][i] for k in v_dict}) for i in range(n)]


def get_ysu_init(grid: scm.grid.StaggeredGrid) -> scm.interfaces.ProgVars:
    """Initial conditions from HND06"""
    z_inv = 500.0  # Inversion height in m

    th = jnp.ones(grid.Nz) * 300.0  # K
    th = jnp.where(grid.z > z_inv, th + 0.01 * (grid.z - z_inv), th)  # linear decrease above inversion

    q = jnp.ones(grid.Nz) * 15.0  # g/kg
    q = jnp.where(grid.z > z_inv, q - 0.01 * (grid.z - z_inv), q)  # linear decrease above inversion up to 1500m
    q = jnp.where(grid.z > 1500, 5.0, q)  # constant above 1500m
    q = q / 1000

    u = jnp.ones(grid.Nz) * 15.0  # m/s
    u = jnp.where(grid.z < z_inv, (15 / 500) * grid.z, u)  # linear increase to 15 m/s at z_inv

    v = jnp.zeros(grid.Nz)

    return scm.interfaces.ProgVars(u=u, v=v, th=th, q=q)


def get_ysu_bc(grid: scm.grid.StaggeredGrid) -> TransientForcing:
    """Boundary conditions from HND06"""

    @jax.jit
    def _shfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface heat flux as function of time in seconds after simulation begin."""
        t_h = t_s / 3600.0  # time in hours
        shfx = jnp.sin((t_h + 2) * jnp.pi / 12) * 400  # W/m2
        shfx = shfx / 1216  # convert to (K m/s)
        return shfx

    @jax.jit
    def _lhfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface latent heat flux as function of time in seconds after simulation begin."""
        t_h = t_s / 3600.0  # time in hours
        lhfx = jnp.sin(t_h * jnp.pi / 12) * 200  # W/m2
        lhfx = lhfx / 1225  # convert to (g/kg m/s)  # todo: correct like this?
        return lhfx

    @jax.jit
    def _u_geo(_) -> jnp.ndarray:
        """Constant geostrophic wind."""
        return jnp.ones(grid.Nz) * 15.0

    @jax.jit
    def _v_geo(_) -> jnp.ndarray:
        """Constant geostrophic wind."""
        return jnp.zeros(grid.Nz)

    return TransientForcing(u_geo=_u_geo, v_geo=_v_geo, f_c=1.39e-4, w_th_s=_shfx, w_q_s=_lhfx)


if __name__ == "__main__":
    # YSU test case
    grid = scm.grid.StaggeredGrid(H=2750, Nz=138)

    ic = get_ysu_init(grid)
    forcing = get_ysu_bc(grid)

    # plot_state(ic, grid)
    #
    # t_h = jnp.linspace(8, 20, 100)
    # shfx = forcing.w_th_s((t_h - 8) * 3600)
    # lhfx = forcing.w_q_s((t_h - 8) * 3600)
    #
    # fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    # ax.plot(t_h, shfx, label="SHFX")
    # ax.plot(t_h, lhfx, label="LHFX")
    # fig.show()

    sfc = SurfaceProperties(z0m=0.1, z0h=0.1, sim_funcs=scm.mo.BusingerDyerSimFuncs(), prescribe="w_th_s")

    # k_mo_closure = create_k_mo_closure(zh=grid.zh, sim_funcs=sfc.sim_funcs)
    model = init_model(grid, sfc, closure_fn=init_ysu_closure(grid=grid))

    state_hist, diag_hist, t = simulate(model, ic, forcing, dt_s=1, t_end_s=45, dt_out_s=5, ode_int="euler")
    # state_hist, diag_hist, t = simulate(
    #     model, ic, forcing, dt_s=0.2, t_end_s=9 * 60 * 60, dt_out_s=60 * 60, ode_int="euler"
    # )
    state_hist = unstack_hist(state_hist)
    diag_hist = unstack_hist(diag_hist)

    # ds = make_dataset(state_hist, diag_hist, time=t, grid=grid)
    # ds.to_netcdf("out.nc")

    plot_hist(state_hist, t, grid, plot_sfc_val=True)
    plot_hist(diag_hist, t, grid, plot_sfc_val=False)
    # plot_sfc_hist(diag_hist, t=t, keys=["u_w", "v_w", "w_th"])
