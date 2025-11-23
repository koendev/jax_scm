from typing import Tuple, Dict

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt


def get_gables1(Nz: int = 128, plot: bool = False, random_seed: int = 0) -> Tuple[Dict, Dict, Dict, Dict]:
    # Grid
    H = 400
    zh = jnp.linspace(0, H, Nz + 1)
    z = 0.5 * (zh[:-1] + zh[1:])
    z_inv = 100

    # Coriolis parameter
    f = 1.39e-4  # 1/s, ~73 deg latitude

    # Geostrophic wind
    ug = jnp.ones_like(z) * 8.0  # m/s
    vg = jnp.zeros_like(z)  # m/s

    # Surface temperature forcing
    th_sfc_0 = 263.5  # K
    th_sfc_fn = lambda t_s: th_sfc_0 - 0.25 * t_s / (60 * 60)  # K, 0.25 K per hour cooling

    # Initial wind profile
    u = jnp.copy(ug).at[0].set(0.0)  # geostrophic wind but with no-slip at surface
    v = jnp.copy(vg)

    # Initial temperature
    th = jnp.ones_like(z) * 265.0  # K
    th = jnp.where(z > z_inv, th + 0.01 * (z - z_inv), th)  # capping inversion
    th = jnp.where(
        z < 50, th + 0.1 * jax.random.normal(key=jax.random.key(random_seed), shape=(Nz,)), th
    )  # random 0.1K perturbation near surface

    # Initial TKE
    tke = jnp.zeros_like(z)
    tke = jnp.where(z < 250, 0.4 * (1 - z / 250) ** 3, tke)  # m^2 s^-2

    # Surface model
    z0m = z0h = 0.1  # m, roughness lengths for momentum and heat
    beta_m = 4.8  # MOST momentum stability coefficent
    beta_h = 7.8  # MOST heat stability coefficient

    grid = {
        "z": z,  # Nz is len
        "zh": zh,  # H is zh[-1]
    }

    forcing = {
        "ug": ug,
        "vg": vg,
        "f": f,
        "th_sfc_0": th_sfc_0,
        "th_sfc_fn": th_sfc_fn,
    }

    init = {
        "u": u,
        "v": v,
        "th": th,
        "tke": tke,
    }

    lsm = {
        "z0m": z0m,
        "z0h": z0h,
        "beta_m": beta_m,
        "beta_h": beta_h,
    }

    if plot:
        # Initial conditions
        fig, axarr = plt.subplots(ncols=3, figsize=(8, 2), sharey="row", layout="constrained")
        axarr[0].plot(u, z, label="u")
        axarr[0].plot(v, z, label="v")
        axarr[0].set_xlabel("Wind (m/s)")
        axarr[0].set_ylabel("Height (m)")
        axarr[0].legend()

        axarr[1].plot(th, z)
        axarr[1].set_xlabel("Potential Temperature (K)")

        axarr[2].plot(tke, z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

        # Forcing
        fig, axarr = plt.subplots(ncols=2, figsize=(8, 2), width_ratios=[1, 3], layout="constrained")
        axarr[0].plot(ug, z, label="ug")
        axarr[0].plot(vg, z, label="vg")
        axarr[0].set_xlabel("Geostrophic Wind (m/s)")
        axarr[0].legend()

        t = jnp.array([0, 9 * 60 * 60])  # 0 and 9 hours
        axarr[1].plot(t, th_sfc_fn(t))
        axarr[1].set_xlabel("Time, s")
        axarr[1].set_ylabel("Surface Potential Temperature (K)")

        fig.show()

    return grid, init, forcing, lsm


def get_ysu():
    # Grid
    Nz = 138
    H = 2750
    z_inv = 500.0  # inversion height in m
    zh = jnp.linspace(0, H, Nz + 1)
    z = 0.5 * (zh[:-1] + zh[1:])

    # Pot temp profile
    th = jnp.ones(Nz) * 300.0  # K
    th = jnp.where(z > z_inv, th + 0.01 * (z - z_inv), th)  # linear decrease above inversion

    # Specific humidity profile
    q = jnp.ones(Nz) * 15.0  # g/kg
    q = jnp.where(z > z_inv, q - 0.01 * (z - z_inv), q)  # linear decrease above inversion up to 1500m
    q = jnp.where(z > 1500, 5.0, q)  # constant above 1500m
    q = q / 1000

    # Wind profile
    u = jnp.ones(Nz) * 15.0  # m/s
    u = jnp.where(z < z_inv, (15 / 500) * z, u)  # linear increase to 15 m/s at z_inv
    v = jnp.zeros(Nz)

    grid = {
        "z": z,  # Nz is len
        "zh": zh,  # H is zh[-1]
    }

    init = {
        "u": u,
        "v": v,
        "th": th,
        "q": q,
    }
    forcing = {}

    return grid, init, forcing


if __name__ == "__main__":
    get_gables1(plot=True)
