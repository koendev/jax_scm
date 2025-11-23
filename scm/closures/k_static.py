from __future__ import annotations

import jax

from scm.interfaces import ProgVars, DiagVars


@jax.jit
def k_static_closure(_, grads: ProgVars) -> DiagVars:
    Km = Kh = 1
    u_w = -Km * grads.u
    v_w = -Km * grads.v
    w_th = -Kh * grads.th

    return DiagVars(u_w=u_w, v_w=v_w, w_th=w_th)
