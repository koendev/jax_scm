from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp

import scm
import scm.consts as consts
from scm.interfaces import DiagVars, ClosureFn, ProgVars


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DiagVarsMOClosure(DiagVars):
    Km: jnp.ndarray  # Horizontal momentum diffusivity
    Kh: jnp.ndarray  # Horizontal heat diffusivity
    L_ob: jnp.ndarray  # Obukhov length scale


def init_mo_closure(zh: jnp.ndarray, sim_funcs: scm.mo.MOSimilarityFuncs) -> ClosureFn:
    phi_m_fn = sim_funcs.get_phi_m_fn()
    phi_h_fn = sim_funcs.get_phi_h_fn()

    @jax.jit
    def _closure(state: ProgVars, grads: ProgVars) -> DiagVarsMOClosure:
        lam0 = 150
        bm, bh = 5, 5

        L = 1 / (1 / (consts.kappa * zh) + 1 / lam0)
        dM_dz = jnp.sqrt(grads.u**2 + grads.v**2)

        th_mean = jnp.zeros_like(zh)
        th_mean = th_mean.at[1:-1].set((state.th[1:] + state.th[:-1]) / 2)
        th_mean = th_mean.at[0].set(state.th[0])  # surface value
        th_mean = th_mean.at[-1].set(state.th[-1])  # top value

        # Init empty fields
        u_w, v_w, w_th, Km, Kh, L_ob = (jnp.zeros_like(zh) for _ in range(6))

        phi_m, phi_h = 1, 1  # initial guess for stability functions
        for _ in range(50):
            Km = L**2 * dM_dz / phi_m**2  # todo: where does this come from?
            Km = jnp.clip(Km, min=0, max=100)

            Kh = L**2 * dM_dz / (phi_m * phi_h)  # todo: where does this come from?
            Kh = jnp.clip(Kh, min=0, max=100)

            u_w = -Km * grads.u
            v_w = -Km * grads.v
            w_th = -Kh * grads.th

            u_st = (u_w**2 + v_w**2) ** 0.25
            L_ob = scm.mo.get_L_obukhov(u_st=u_st, w_th=w_th, th=th_mean)
            zeta = zh / L_ob
            zeta = jnp.clip(zeta, -10, 10)

            # Update stability functions
            phi_m = phi_m_fn(zeta)
            phi_h = phi_h_fn(zeta)

        return DiagVarsMOClosure(u_w=u_w, v_w=v_w, w_th=w_th, Km=Km, Kh=Kh, L_ob=L_ob)

    return _closure
