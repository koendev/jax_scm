import jax.numpy as jnp


def get_ml_th_inversion(th_ml: float, th_inv_rate: float, z: jnp.ndarray, z_inv: float) -> jnp.ndarray:
    """Return inversion profile for potential temperature."""
    theta = jnp.ones_like(z) * th_ml
    theta = theta.at[z > z_inv].set(th_ml + th_inv_rate * (z[z > z_inv] - z_inv))
    return theta
