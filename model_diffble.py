import functools
from typing import Tuple, Protocol

import jax
import jax.numpy as jnp
import optax
import tqdm
from flax import nnx

from odeint import init_time_stepper
from scm.grid import StaggeredGrid
from scm.interfaces import DiagVars, ModelFn, ProgVars
from scm.nn import FlexMLP

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


class ClosureFn(Protocol):
    def __call__(self, state: ProgVars, grads: ProgVars, **kwargs) -> DiagVars:
        """Compute closure variables (fluxes and variances) from state and gradients"""


@nnx.jit
def nn_closure(state: ProgVars, grads: ProgVars, nn: nnx.Module) -> DiagVars:
    Km, Kh = nn(grads.as_tensor().T).T
    u_w = -Km * grads.u
    v_w = -Km * grads.v
    w_th = -Kh * grads.th

    return DiagVars(u_w=u_w, v_w=v_w, w_th=w_th)


@jax.jit
def k_static_closure(_, grads: ProgVars) -> DiagVars:
    Km = Kh = 1
    u_w = -Km * grads.u
    v_w = -Km * grads.v
    w_th = -Kh * grads.th

    return DiagVars(u_w=u_w, v_w=v_w, w_th=w_th)


def create_scm(grid: StaggeredGrid, closure_fn: ClosureFn) -> ModelFn:
    f = 1e-4
    u_geo, v_geo = 8, 0  # Geostrophic wind components

    @nnx.jit
    def _model(state: ProgVars, nn: nnx.Module) -> Tuple[ProgVars, DiagVars]:
        # Unpack state
        u, v, th = state.u, state.v, state.th

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = jnp.zeros(grid.Nz_h)
        du_dz = du_dz.at[1:-1].set((u[1:] - u[:-1]) / grid.dz)
        # du_dz = du_dz.at[0].set(du_dz_s)

        dv_dz = jnp.zeros(grid.Nz_h)
        dv_dz = dv_dz.at[1:-1].set((v[1:] - v[:-1]) / grid.dz)
        # dv_dz = dv_dz.at[0].set(dv_dz_s)

        dth_dz = jnp.zeros(grid.Nz_h)
        dth_dz = dth_dz.at[1:-1].set((th[1:] - th[:-1]) / grid.dz)
        # dth_dz = dth_dz.at[0].set(dth_dz_s)

        # PBL SCHEME on half levels
        grads = ProgVars(u=du_dz, v=dv_dz, th=dth_dz)
        fluxes = closure_fn(state, grads, nn)
        u_w, v_w, w_th = fluxes.u_w, fluxes.v_w, fluxes.w_th  # unpack

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_th = (w_th[1:] - w_th[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f * v - f * v_geo - div_u_w
        v_tend = -f * u + f * u_geo - div_v_w
        th_tend = -div_w_th

        # Gather tendencies and diagnostics
        tends = ProgVars(u=u_tend, v=v_tend, th=th_tend)
        diag = DiagVars(u_w=u_w, v_w=v_w, w_th=w_th)
        return tends, diag

    return _model


@functools.partial(nnx.jit, static_argnames=("scm_stepper"))
def train_step(nn, optimizer, scm_stepper, x: ProgVars, x_next_true: ProgVars):
    def loss_fn(nn):
        # todo: proper ode integrator
        # todo: scaling!
        x_next_pred, _ = scm_stepper(x, nn=nn)

        return jnp.mean((x_next_pred.as_tensor() - x_next_true.as_tensor()) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(nn)
    optimizer.update(grads)

    return loss


if __name__ == "__main__":
    # Setup nn
    nn = FlexMLP([3, 8, 2], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(nn, optax.adam(1e-1))  # reference sharing

    # Setup SCM
    grid = StaggeredGrid(H=400, Nz=16)
    scm = create_scm(grid, closure_fn=nn_closure)
    scm_stepper = init_time_stepper(scm, dt=0.1, method="euler")
    # scm = wrap_dicts(scm)

    # State
    init = ProgVars(u=jnp.ones(grid.Nz) * grid.z / grid.H * 1, v=jnp.zeros(grid.Nz), th=jnp.ones(grid.Nz) * 1)
    target = ProgVars(u=jnp.zeros(grid.Nz), v=jnp.zeros(grid.Nz), th=jnp.zeros(grid.Nz))

    for _ in (pbar := tqdm.trange(100)):
        loss = train_step(nn, optimizer, scm_stepper, init, target)
        pbar.set_description(f"loss={loss}")
