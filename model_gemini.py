import jax
import jax.numpy as jnp
from typing import NamedTuple


import cases

# jax.config.update("jax_disable_jit", True)


# --- 1. Configuration and Constants ---
class Constants(NamedTuple):
    g: float = 9.81
    kappa: float = 0.40
    Cp: float = 1004.0
    rho: float = 1.225
    f: float = 1e-4
    z0: float = 0.1
    lambda_0: float = 40.0
    Pr_t: float = 1.0
    C_k: float = 0.1
    C_e: float = 0.93


class Grid(NamedTuple):
    dz: float
    nz: int
    z_center: jnp.ndarray
    z_face: jnp.ndarray


class State(NamedTuple):
    u: jnp.ndarray
    v: jnp.ndarray
    theta: jnp.ndarray
    tke: jnp.ndarray
    time: float


class Diagnostics(NamedTuple):
    ustar: jnp.ndarray
    wt_sfc: jnp.ndarray
    Km: jnp.ndarray
    Kh: jnp.ndarray


# --- 2. Differentiable Physics ---
@jax.jit
def blackadar_mixing_length(z, const: Constants):
    l = (const.kappa * z) / (1.0 + (const.kappa * z) / const.lambda_0)
    return l


@jax.jit
def calc_diffusivity(tke, l_mix, const: Constants):
    # FIX: Use softplus for differentiability near zero
    # softplus(x) = log(1 + exp(x)). Smooth approximation of ReLU.
    # We shift it slightly so softplus(0) isn't exactly involved if not needed,
    # but fundamentally this prevents sqrt(negative) and keeps gradients alive.
    tke_safe = jax.nn.softplus(tke)

    Km = const.C_k * l_mix * jnp.sqrt(tke_safe)
    Kh = Km / const.Pr_t
    return Km, Kh


@jax.jit
def psi_m(zeta):
    """
    Integrated stability function for momentum.
    FIX: Protected against NaNs in the inactive branch of jnp.where.
    """
    # 1. Protect the unstable calculation inputs
    # When zeta > 0, this branch is not used, but we must ensure the math is valid
    # to avoid NaNs in the computation graph.
    zeta_unstable = jnp.minimum(zeta, -1e-8)
    x = (1.0 - 16.0 * zeta_unstable) ** 0.25

    psi_unstable = 2.0 * jnp.log((1.0 + x) / 2.0) + jnp.log((1.0 + x**2) / 2.0) - 2.0 * jnp.arctan(x) + jnp.pi / 2.0

    # 2. Protect the stable calculation inputs
    zeta_stable = jnp.maximum(zeta, 0.0)
    psi_stable = -5.0 * zeta_stable

    return jnp.where(zeta < 0, psi_unstable, psi_stable)


@jax.jit
def psi_h(zeta):
    """
    Integrated stability function for heat.
    FIX: Protected against NaNs in the inactive branch.
    """
    zeta_unstable = jnp.minimum(zeta, -1e-8)
    x = (1.0 - 16.0 * zeta_unstable) ** 0.25
    psi_unstable = 2.0 * jnp.log((1.0 + x**2) / 2.0)

    zeta_stable = jnp.maximum(zeta, 0.0)
    psi_stable = -5.0 * zeta_stable

    return jnp.where(zeta < 0, psi_unstable, psi_stable)


@jax.jit
def surface_layer_fluxes(u1, v1, theta1, theta_sfc, z1, const: Constants):
    wind_speed = jnp.sqrt(u1**2 + v1**2 + 1e-4)  # Added epsilon inside sqrt
    delta_theta = theta1 - theta_sfc

    # Initial guess (Neutral)
    ustar_guess = const.kappa * wind_speed / jnp.log(z1 / const.z0)
    tstar_guess = const.kappa * delta_theta / jnp.log(z1 / const.z0)

    def body_fn(i, val):
        u_s, t_s = val

        # FIX: Smooth Regularization instead of branching for small t_s
        # This prevents division by zero while keeping the gradient smooth.
        # sign(t_s) * max(|t_s|, 1e-5)
        t_s_sign = jnp.sign(t_s)
        # If sign is 0, assume positive
        t_s_sign = jnp.where(t_s_sign == 0, 1.0, t_s_sign)
        t_s_safe = t_s_sign * jnp.maximum(jnp.abs(t_s), 1e-5)

        L = (u_s**2 * theta1) / (const.kappa * const.g * t_s_safe)

        zeta = z1 / L
        zeta = jnp.clip(zeta, -5.0, 5.0)

        # Update ustar
        denom_m = jnp.log(z1 / const.z0) - psi_m(zeta)
        u_new = (const.kappa * wind_speed) / denom_m

        # Update tstar
        denom_h = jnp.log(z1 / const.z0) - psi_h(zeta)
        t_new = (const.kappa * delta_theta) / denom_h

        return (u_new, t_new)

    # Note: Gradients will flow through these 5 unrolled iterations
    ustar_final, tstar_final = jax.lax.fori_loop(0, 10, body_fn, (ustar_guess, tstar_guess))

    uw_sfc = -(ustar_final**2)
    wt_sfc = -ustar_final * tstar_final

    return uw_sfc, wt_sfc, ustar_final


# --- 3. Numerical Solvers ---
def solve_tridiagonal_diffusion(phi, K_face, source_term, dt, dz, bc_flux_bot, bc_flux_top=0.0):
    """
    Solves diffusion equation using jax.lax.linalg.tridiagonal_solve.
    Fixes shapes for strict rank requirements.
    """
    N = phi.shape[0]
    alpha = dt / (2.0 * dz**2)

    # K_face has N+1 elements.
    # K_lower refers to interfaces i-1/2 (indices 0 to N-1 in K_face)
    # K_upper refers to interfaces i+1/2 (indices 1 to N in K_face)
    K_lower = K_face[:-1]
    K_upper = K_face[1:]

    # Construct diagonals (All size N)
    main_diag = 1.0 + alpha * (K_lower + K_upper)
    lower_diag = -alpha * K_lower
    upper_diag = -alpha * K_upper

    # --- Explicit Part (RHS) ---
    flux_diff = K_face[1:-1] * (phi[1:] - phi[:-1]) / dz
    flux_all = jnp.concatenate([jnp.array([bc_flux_bot]), flux_diff, jnp.array([bc_flux_top])])
    diff_term = (flux_all[1:] - flux_all[:-1]) / dz

    rhs = phi + dt * (0.5 * diff_term + source_term)

    # --- Boundary Conditions ---
    # Modify Main Diagonals
    main_diag = main_diag.at[0].set(1.0 + alpha * K_face[1])
    main_diag = main_diag.at[-1].set(1.0 + alpha * K_face[-2])

    # Cut off connections to ghost cells (crucial for correctness)
    # In 'lax', we simply set the coefficient to 0.0.
    # Note: lax ignores dl[0] and du[-1] anyway, but we zero the neighbors
    # to enforce the Neumann BC logic explicitly.
    lower_diag = lower_diag.at[1].set(0.0)  # No influence from i=-1 on i=0
    upper_diag = upper_diag.at[-2].set(0.0)  # No influence from i=N on i=N-1

    # Add Flux BC forcing to RHS
    rhs = rhs.at[0].add((dt / dz) * bc_flux_bot)
    rhs = rhs.at[-1].add(-(dt / dz) * bc_flux_top)

    # --- FIX FOR JAX.LAX ---

    # 1. Expand RHS dims: (N,) -> (N, 1)
    rhs = rhs[:, None]

    # 2. Call lax solver with full-length diagonals
    # lax.tridiagonal_solve expects (dl, d, du, b)
    # dl, d, du must all be shape (..., N)
    phi_new = jax.lax.linalg.tridiagonal_solve(lower_diag, main_diag, upper_diag, rhs)

    # 3. Squeeze back to (N,)
    return phi_new.squeeze()


# --- 4. Time Stepping ---
def time_step_fn(state: State, inputs):
    grid, const, forcing, dt = inputs
    u_g, v_g, theta_sfc = forcing

    # 1. Diagnostics
    tke_face = 0.5 * (state.tke[:-1] + state.tke[1:])
    tke_face = jnp.pad(tke_face, (1, 1), mode="edge")

    l_mix = blackadar_mixing_length(grid.z_face, const)
    Km, Kh = calc_diffusivity(tke_face, l_mix, const)

    # 2. Surface Fluxes
    u1, v1, theta1 = state.u[0], state.v[0], state.theta[0]
    z1 = grid.z_center[0]
    uw_sfc, wt_sfc, ustar = surface_layer_fluxes(u1, v1, theta1, theta_sfc, z1, const)

    wspd = jnp.sqrt(u1**2 + v1**2 + 1e-6)
    uw_sfc_x = uw_sfc * (u1 / wspd)
    vw_sfc_y = uw_sfc * (v1 / wspd)

    # 3. TKE Prognostic
    du_dz = (state.u[1:] - state.u[:-1]) / grid.dz
    dv_dz = (state.v[1:] - state.v[:-1]) / grid.dz
    shear_sq = du_dz**2 + dv_dz**2
    shear_sq_cen = jnp.pad(shear_sq, (0, 1), mode="edge")
    Km_cen = 0.5 * (Km[:-1] + Km[1:])
    shear_prod = Km_cen * shear_sq_cen

    dtheta_dz = (state.theta[1:] - state.theta[:-1]) / grid.dz
    dtheta_dz_cen = jnp.pad(dtheta_dz, (0, 1), mode="edge")
    Kh_cen = 0.5 * (Kh[:-1] + Kh[1:])
    buoyancy = -(const.g / 300.0) * Kh_cen * dtheta_dz_cen

    l_mix_cen = blackadar_mixing_length(grid.z_center, const)

    # FIX: Use softplus for TKE in dissipation term to ensure gradients flow
    tke_safe_cen = jax.nn.softplus(state.tke)
    dissipation = const.C_e * (tke_safe_cen**1.5) / l_mix_cen

    tke_source = shear_prod + buoyancy - dissipation

    new_tke = solve_tridiagonal_diffusion(state.tke, Km, tke_source, dt, grid.dz, 0.0, 0.0)

    # 4. Momentum
    u_source = const.f * (state.v - v_g)
    v_source = -const.f * (state.u - u_g)

    new_u = solve_tridiagonal_diffusion(state.u, Km, u_source, dt, grid.dz, uw_sfc_x, 0.0)
    new_v = solve_tridiagonal_diffusion(state.v, Km, v_source, dt, grid.dz, vw_sfc_y, 0.0)

    # 5. Thermodynamics
    new_theta = solve_tridiagonal_diffusion(state.theta, Kh, jnp.zeros_like(state.theta), dt, grid.dz, wt_sfc, 0.0)

    new_state = State(u=new_u, v=new_v, theta=new_theta, tke=new_tke, time=state.time + dt)
    new_diags = Diagnostics(
        ustar=ustar,
        wt_sfc=wt_sfc,
        Km=Km_cen,
        Kh=Kh_cen,
    )
    return new_state, new_diags


# --- 5. Differentiability Check ---
def run_differentiability_check():
    # Setup simple grid
    nz = 20
    H = 1000.0
    dz = H / nz
    z_face = jnp.linspace(0, H, nz + 1)
    z_center = 0.5 * (z_face[:-1] + z_face[1:])
    grid = Grid(dz=dz, nz=nz, z_center=z_center, z_face=z_face)
    const = Constants()
    dt = 10.0

    # Wrapper to differentiate: Loss = Final Surface Friction Velocity
    # We want to see if changing initial wind changes final u_star
    def simulation_loss(u_initial_guess):
        u_init = u_initial_guess * jnp.ones(nz)
        v_init = jnp.zeros(nz)
        theta_init = 300.0 * jnp.ones(nz)
        tke_init = jnp.ones(nz) * 0.1

        state = State(u=u_init, v=v_init, theta=theta_init, tke=tke_init, time=0.0)
        forcing = (10.0, 0.0, 300.0)  # u_g, v_g, theta_sfc

        # Run 10 steps
        def step_wrapper(s, _):
            return time_step_fn(s, (grid, const, forcing, dt))

        final_state, history = jax.lax.scan(step_wrapper, state, None, length=10)

        # Loss function: Try to maximize TKE (just a dummy objective)
        # or simply return a value to check gradient
        return jnp.sum(final_state.tke)

    # Compute Value and Gradient
    initial_wind_speed = 5.0
    loss_val, grad_val = jax.value_and_grad(simulation_loss)(initial_wind_speed)

    print(f"Loss Value: {loss_val}")
    print(f"Gradient w.r.t Initial Wind: {grad_val}")

    if jnp.isnan(grad_val):
        print("FAIL: Gradient is NaN!")
    else:
        print("SUCCESS: Gradient is finite and calculated.")


# --- 6. Main Simulation Loop ---
def run_simulation():
    grid, init, forcing, lsm = cases.get_gables1()

    # Grid Setup
    grid = Grid(dz=jnp.diff(grid["z"])[0], nz=len(grid["z"]), z_center=grid["z"], z_face=grid["zh"])

    const = Constants()

    # Initial Conditions
    state = State(
        u=init["u"],
        v=init["v"],
        theta=init["th"],
        tke=init["tke"],
        time=0.0,
    )

    # Forcing
    u_g = forcing["ug"]
    v_g = forcing["vg"]
    theta_sfc = forcing["th_sfc_0"]
    forcing = (u_g, v_g, theta_sfc)

    dt = 10.0  # seconds
    simulation_time = 60 * 60 * 9  # 6 hours
    n_steps = int(simulation_time / dt)

    # Scan Loop
    # We use a wrapper to pass static args (grid, const) and constant forcing
    def step_wrapper(state, _):
        new_state, diags = time_step_fn(state, (grid, const, forcing, dt))
        return new_state, (new_state, diags)

    print("JIT compiling and running simulation...")
    final_state, history = jax.lax.scan(step_wrapper, state, None, length=n_steps)

    return grid, history


if __name__ == "__main__":
    import xarray as xr

    # run_differentiability_check()
    grid, history = run_simulation()
    (states, diags) = history
    ustar_hist = diags[0]

    print("Simulation complete.")
    print(f"Final Surface Friction Velocity (u*): {ustar_hist[-1]:.4f} m/s")
    print(f"Final TKE at lowest level: {states.tke[-1, 0]:.4f} m2/s2")

    # Optional: Simple ASCII Plot of final Theta profile
    print("\nFinal Potential Temperature Profile (Lowest 10 levels):")
    for k in range(10):
        print(f"Z={grid.z_center[k]:.1f}m : Theta={states.theta[-1, k]:.4f} K")

    ds = xr.Dataset(
        {
            "u": (("time", "z"), history[0].u),
            "v": (("time", "z"), history[0].v),
            "theta": (("time", "z"), history[0].theta),
            "tke": (("time", "z"), history[0].tke),
            "ustar": (("time",), history[1].ustar),
            "wt_sfc": (("time",), history[1].wt_sfc),
            "Km": (("time", "z"), history[1].Km),
            "Kh": (("time", "z"), history[1].Kh),
        },
        coords={
            "z": grid.z_center,
        },
    )
    ds.to_netcdf("out.nc")
