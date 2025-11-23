import jax.numpy as jnp


def f_c_from_lat(lat_deg: float) -> float:
    """Calculate the Coriolis parameter from latitude in degrees."""
    return float(2 * 7.2921e-5 * jnp.sin(jnp.radians(lat_deg)))
