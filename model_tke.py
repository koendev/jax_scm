from typing import Tuple, Type

import jax
from jax import numpy as jnp

import scm.consts
import scm.mo
from scm.grid import StaggeredGrid
from scm.interfaces import DiagVars, ModelFn, ProgVars


# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


class ProgVarsMY15(ProgVars):
    """1.5-order prognostic variables"""

    tke: jnp.ndarray  # Turbulent kinetic energy
    th_var: jnp.ndarray  # Potential temperature variance


class DiagVarsMY15(DiagVars):
    """1.5-order diagnosed variables"""

    Km: jnp.ndarray
    Kh: jnp.ndarray
    eps: jnp.ndarray
    eps_th: jnp.ndarray


def create_model(grid: StaggeredGrid) -> ModelFn:
    # Km = Kh = 1
    f = 1e-4  # Coriolis parameter, s^-1
    u_geo = 8
    v_geo = 0

    mo_fn = scm.mo.init_mo_sfc(
        z0m=0.1,
        z0h=0.01,
        z=grid.dz / 2,
        sim_funcs=scm.mo.BusingerDyerSimFuncs(),
        prescribe="w_th_s",
    )

    # Emperical length scales
    lam = lam1 = lam2 = lam3 = lam4 = 1

    @jax.jit
    def _interp_full_half(x: jnp.ndarray, x_bottom: jnp.ndarray, x_top: jnp.ndarray) -> jnp.ndarray:
        """Interpolate a full-level variable to half levels."""
        x_h = jnp.zeros(grid.Nz_h)
        x_h = x_h.at[1:-1].set((x[1:] + x[:-1]) / 2)
        x_h = x_h.at[0].set(x_bottom)
        x_h = x_h.at[-1].set(x_top)
        return x_h

    @jax.jit
    def _avg_half_full(x_h: jnp.ndarray) -> jnp.ndarray:
        """Average a half-level variable to full levels."""
        return (x_h[1:] + x_h[:-1]) / 2

    @jax.jit
    def _model(state: ProgVarsMY15) -> Tuple[ProgVarsMY15, DiagVarsMY15]:
        # Get prognostic variables for easier access
        u, v, th, tke, th_var = state["u"], state["v"], state["th"], state["tke"], state["th_var"]

        # Interpolate tke to half levels
        tke_h = _interp_full_half(tke, tke[0], tke[-1])  # todo: surface value MOST?

        # Get surface variables from MO model
        u_st, w_th_s, L, du_dz_s, dv_dz_s, dth_dz_s, m10, th2, th_s, u_w_s, v_w_s = mo_fn(
            u_0=u[0], v_0=v[0], th_0=th[0], w_th_s=+0.05
        )

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = jnp.zeros(grid.Nz_h)
        du_dz = du_dz.at[1:-1].set((u[1:] - u[:-1]) / grid.dz)
        du_dz = du_dz.at[0].set(du_dz_s)

        dv_dz = jnp.zeros(grid.Nz_h)
        dv_dz = dv_dz.at[1:-1].set((v[1:] - v[:-1]) / grid.dz)
        dv_dz = dv_dz.at[0].set(dv_dz_s)

        dth_dz = jnp.zeros(grid.Nz_h)
        dth_dz = dth_dz.at[1:-1].set((th[1:] - th[:-1]) / grid.dz)
        dth_dz = dth_dz.at[0].set(dth_dz_s)

        # todo: surface value?!
        dtke_dz = jnp.zeros(grid.Nz_h)
        dtke_dz = dtke_dz.at[1:-1].set((tke[1:] - tke[:-1]) / grid.dz)

        # todo: surface value?!
        dth_var_dz = jnp.zeros(grid.Nz_h)
        dth_var_dz = dth_var_dz.at[1:-1].set((th_var[1:] - th_var[:-1]) / grid.dz)

        # Compute eddy diffusivities (half levels)
        Km = lam * tke_h ** (-1 / 2)
        Kh = 1.35 * Km

        # Compute momentum fluxes with lowest level set to MOST (half levels)
        u_w = -Km * du_dz
        u_w = u_w.at[0].set(u_w_s)

        v_w = -Km * dv_dz
        v_w = v_w.at[0].set(v_w_s)

        w_th = -Kh * dth_dz  # todo: -gamma_c for non-local mixing
        w_th = w_th.at[0].set(w_th_s)

        # Compute higher-order terms (half levels)
        w_p_tke = (5 / 3) * lam4 * tke_h ** (-1 / 2) * dtke_dz
        w_th_th = lam3 * tke_h ** (-1 / 2) * dth_var_dz

        # Compute dissipation rates (full levels bc directly into prognostic equations)
        eps_r = 0
        eps = tke ** (3 / 2) / lam1
        eps_th = tke ** (1 / 2) * th_var / lam2

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_th = (w_th[1:] - w_th[:-1]) / grid.dz
        div_w_p_tke = (w_p_tke[1:] - w_p_tke[:-1]) / grid.dz
        div_w_th_th = (w_th_th[1:] - w_th_th[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f * v - f * v_geo - div_u_w
        v_tend = -f * u + f * u_geo - div_v_w
        th_tend = -div_w_th

        tke_tend = -_avg_half_full(u_w * du_dz) - _avg_half_full(v_w * dv_dz)  # shear production
        tke_tend = tke_tend + (scm.consts.g / th) * _avg_half_full(w_th)  # buoyancy
        tke_tend = tke_tend - div_w_p_tke - eps  # transport and dissipation

        th_var_tend = -_avg_half_full(2 * w_th * dth_dz)  # buoyancy
        th_var_tend = th_var_tend - div_w_th_th - 2 * eps_th - eps_r  # transport, dissipation, and relaxation

        # Gather tendencies and diagnosed variables
        tends = ProgVarsMY15(u=u_tend, v=v_tend, th=th_tend, tke=tke_tend, th_var=th_var_tend)
        diag = DiagVarsMY15(u_w=u_w, v_w=v_w, w_th=w_th, Km=Km, Kh=Kh, eps=eps, eps_th=eps_th)
        return tends, diag

    return _model


def simulate(model, state_init: ProgVarsMY15, dt: float = 60.0, steps: int = 100) -> Tuple[ProgVarsMY15, DiagVarsMY15]:
    prog_vars_type: Type[ProgVars] = ProgVarsMY15

    @jax.jit
    def _euler_step(state: ProgVars, _) -> Tuple[ProgVars, Tuple[ProgVars, DiagVars]]:
        tends, diag = model(state)
        state_new = prog_vars_type(**{k: state[k] + tends[k] * dt for k in prog_vars_type.__annotations__.keys()})
        return state_new, (state_new, diag)

    _, (state_traj, diag_traj) = jax.lax.scan(_euler_step, init=state_init, length=steps)
    return state_traj, diag_traj


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid = StaggeredGrid(H=1000, Nz=16)

    state_init = ProgVarsMY15(
        u=jnp.ones(grid.Nz) * 10,
        v=jnp.zeros(grid.Nz),
        th=jnp.ones(grid.Nz) * 300,
        tke=jnp.ones(grid.Nz) * 0.1,  # todo: on half levels
        th_var=jnp.ones(grid.Nz) * 0.01,  # todo: on half levels
    )
    model = create_model(grid)
    (state_traj, diag_traj) = simulate(model, state_init, dt=0.01, steps=int(1 * 60 / 0.01))

    fig, axarr = plt.subplots(ncols=len(state_traj), figsize=(8, 6))
    for ax, (var_name, var_data) in zip(axarr, state_traj.items()):
        ax.plot(var_data[-1], grid.z, label=var_name)
        ax.set_title(var_name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Height (m)")
        ax.legend()
    fig.show()
