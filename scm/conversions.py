import jax
import jax.numpy as jnp


@jax.jit
def thv_to_th(*, thv: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Virtual potential temperature to dry potential temperature."""
    return thv / (1 + 0.61 * qv)


@jax.jit
def th_to_thv(*, th: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Dry potential temperature to virtual potential temperature."""
    return th * (1 + 0.61 * qv)


@jax.jit
def w_th_to_w_thv(*, th: jnp.ndarray, w_th: jnp.ndarray, w_qv: jnp.ndarray) -> jnp.ndarray:
    """Sensible heat flux (w'theta') to buoyancy flux (w'theta_v')."""
    return w_th + 0.61 * th * w_qv
