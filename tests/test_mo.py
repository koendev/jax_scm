import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scm.mo import init_mo_sfc, BusingerDyerSimFuncs, get_L_obukhov, MOResult
from scm import consts
import matplotlib.pyplot as plt


# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


# Set up common test fixtures
@pytest.fixture
def bd_mo_w_th_s():
    """Businger-Dyer model with prescribed surface heat flux"""
    return init_mo_sfc(z0m=0.1, z0h=0.01, z=5, z_grad=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")


@pytest.fixture
def bd_mo_th_s():
    """Businger-Dyer model with prescribed surface temp"""
    return init_mo_sfc(z0m=0.1, z0h=0.01, z=5, z_grad=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="th_s")


@pytest.fixture(params=[False, True])
def use_jit(request):
    """Fixture to run tests twice -- with JIT enabled and disabled.
    Start with False for debugging purposes.
    """
    # Store the original JIT setting to restore later
    original = jax.config.jax_disable_jit

    # Set JIT state based on parameter (note: jax_disable_jit is inverted)
    jax.config.update("jax_disable_jit", not request.param)

    # Yield the current JIT enabled state to the test
    yield request.param

    # Restore original setting after test completes
    jax.config.update("jax_disable_jit", original)


def test_bd(use_jit):
    """Test that BD functions can be called without errors."""
    phi_m_fn, phi_h_fn, psi_m_fn, psi_h_fn = BusingerDyerSimFuncs().get_all_fns()
    for zeta in jnp.array([-1.0, 0.0, 1.0]):
        phi_m_fn(zeta)
        phi_h_fn(zeta)
        psi_m_fn(zeta)
        psi_h_fn(zeta)


def test_computation():
    pytest.skip("Only outdated plotting")
    # Constants
    z0m, z0h, dz0 = 0.1, 0.01, 10
    z = dz0 / 2

    # Take know fluxes create testing mesh
    th_0 = 290.0  # Pot temp at first level
    u_st = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # Friction velocity in m/s
    # w_th = [-0.15, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05]  # Surface heat flux in K m/s
    # w_th = [0.15, 0.1, 0.05, 0.025, 0.0, -0.025, -0.05]  # Surface heat flux in K m/s
    w_th = jnp.arange(-0.15, 0.2, 0.05)

    u_st, w_th = jnp.meshgrid(jnp.array(u_st), jnp.array(w_th), indexing="ij")
    u_st = u_st.flatten()
    w_th = w_th.flatten()

    # Compute Obukhov length and drop cases where L is inf
    L = get_L_obukhov(u_st, w_th, th_0)

    valid_mask = jnp.isfinite(L)
    u_st = u_st[valid_mask]
    w_th = w_th[valid_mask]
    L = L[valid_mask]

    # Compute theta_star
    th_st = -w_th / u_st

    # Use Businger-Dyer simulation functions to compute gradients, which will serve as input to the test
    bd = BusingerDyerSimFuncs()
    phi_m_fn, phi_h_fn, psi_m_fn, psi_h_fn = bd.get_all_fns()

    # Compute expected velocity and surface temperature from known fluxes
    u0 = u_st / consts.kappa * (jnp.log(z / z0m) - psi_m_fn(z / L) + psi_m_fn(z0m / L))
    th_s = th_0 - th_st / consts.kappa * (jnp.log(z / z0h) - psi_h_fn(z / L) + psi_h_fn(z0h / L))

    # Compute expected gradients
    du_dz = u_st / (consts.kappa * z) * phi_m_fn(z / L)
    dth_dz = th_st / (consts.kappa * z) * phi_h_fn(z / L)

    # Solve for the expected outputs
    bd_mo = init_mo_sfc(z0m=z0m, z0h=z0h, dz0=dz0, sim_funcs=bd, prescribe="w_th_s")
    res = [bd_mo(u_0=u0_i, v_0=0.0, th_0=th_0, w_th_s=w_th_i) for u0_i, w_th_i in zip(u0, w_th)]
    res = jnp.array(res).T
    u_st_res, _, L_res, du_dz_res, _, dth_dz_res, _, _, th_s_res = res

    # Create mapping between expected and computed values
    mapping = [
        (u_st, u_st_res, "u_st"),
        (th_s, th_s_res, "th_s"),
        (L, L_res, "L"),
        (du_dz, du_dz_res, "du_dz"),
        (dth_dz, dth_dz_res, "dth_dz"),
        (th_st, th_s_res, "th_st"),
    ]

    # Plot the results
    ncols = len(mapping) + 1
    fig, axarr = plt.subplots(ncols=ncols, figsize=(3 * ncols, 3))
    for ax, (v_exp, v_comp, label) in zip(axarr.flat[1:], mapping):
        # Use error for coloring
        err = jnp.abs(v_exp - v_comp) < 1e-3
        color = np.where(err, "green", "red")

        ax.scatter(v_exp, v_comp, s=1, c=color, alpha=0.5)
        ax.set_xlabel("expected")
        ax.set_ylabel("computed")
        ax.set_title(label)

    # Plot the scatter of u_st vs w_th
    err = jnp.abs(u_st - u_st_res) < 1e-3
    color = np.where(err, "green", "red")
    ax = axarr.flat[0]
    ax.scatter(u_st, w_th, s=1, color=color)

    fig.show()

    # Asserts to make test work
    for v_exp, v_comp, label in mapping:
        assert jnp.allclose(v_exp, v_comp, rtol=1e-5), f"{label}: Mismatch in expected vs result"


def test_neutral_conditions(bd_mo_w_th_s, use_jit):
    """Test near-neutral conditions where w_th_s ≈ 0"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.0)

    # In neutral conditions, L should be very large
    assert jnp.abs(res.L) > 1000
    # Friction velocity should be positive
    assert res.u_st > 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, 0.0)
    # dth_dz should be close to zero for neutral conditions
    assert jnp.abs(res.dth_dz) < 0.01


def test_unstable_conditions(bd_mo_w_th_s, use_jit):
    """Test unstable conditions with positive heat flux"""
    res: MOResult = bd_mo_w_th_s(u_0=3.0, v_0=0.0, th_0=290.0, w_th_s=0.1)

    # For unstable conditions, L should be negative
    assert res.L < 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, 0.1)
    # Negative temperature gradient in unstable conditions
    assert res.dth_dz < 0


def test_stable_conditions(bd_mo_w_th_s, use_jit):
    """Test stable conditions with negative heat flux"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=-0.01)

    # For stable conditions, L should be positive
    assert res.L > 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, -0.01)
    # Positive temperature gradient in stable conditions
    assert res.dth_dz > 0


def test_wind_direction(bd_mo_w_th_s, use_jit):
    """Test that the model handles different wind directions correctly"""
    # Wind from east
    res_east = bd_mo_w_th_s(u_0=3.0, v_0=0.0, th_0=290.0, w_th_s=0.01)
    # Wind from north
    res_north = bd_mo_w_th_s(u_0=0.0, v_0=3.0, th_0=290.0, w_th_s=0.01)
    # Wind from northeast (same magnitude)
    u_ne = v_ne = 3.0 / jnp.sqrt(2)
    res_ne = bd_mo_w_th_s(u_0=u_ne, v_0=v_ne, th_0=290.0, w_th_s=0.01)

    # Friction velocity should be similar for same wind speed magnitude
    assert jnp.isclose(res_east.u_st, res_north.u_st, rtol=1e-5)
    assert jnp.isclose(res_east.u_st, res_ne.u_st, rtol=1e-5)


def test_wind_magnitude(bd_mo_w_th_s, use_jit):
    """Test different wind magnitudes"""
    # Light wind
    res_light: MOResult = bd_mo_w_th_s(u_0=1.0, v_0=0.0, th_0=290.0, w_th_s=0.01)
    # Moderate wind
    res_mod: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.01)
    # Strong wind
    res_strong: MOResult = bd_mo_w_th_s(u_0=10.0, v_0=0.0, th_0=290.0, w_th_s=0.01)

    # Friction velocity should increase with wind speed
    assert res_light.u_st < res_mod.u_st < res_strong.u_st
    # L should increase with wind speed for same heat flux (unstable)
    assert jnp.abs(res_light.L) < jnp.abs(res_mod.L) < jnp.abs(res_strong.L)


def test_gradient_directions(bd_mo_w_th_s, use_jit):
    """Test that gradients have the right sign for different stability conditions"""
    # Unstable case
    res_unst: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.1)

    # Stable case
    res_stab: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=-0.1)

    # In unstable conditions, velocity gradients are smaller than in stable conditions
    assert jnp.abs(res_unst.du_dz) < jnp.abs(res_stab.du_dz)
    # Temperature gradient is negative in unstable, positive in stable
    assert res_unst.dth_dz < 0
    assert res_stab.dth_dz > 0


def test_extreme_conditions(use_jit):
    """Test extreme conditions that might challenge the iteration scheme"""
    bd_mo_w_th_s = init_mo_sfc(z0m=0.1, z0h=0.01, z=5, z_grad=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")

    # Very strong instability (large positive heat flux, low wind)
    res_unst: MOResult = bd_mo_w_th_s(u_0=0.5, v_0=0.0, th_0=290.0, w_th_s=0.3)

    # Should converge to a reasonable L value
    assert not jnp.isnan(res_unst.L)
    assert res_unst.L < 0  # Unstable

    # Very strong stability (large negative heat flux)
    res_stab = bd_mo_w_th_s(u_0=0.5, v_0=0.0, th_0=290.0, w_th_s=-0.3)

    # Should converge to a reasonable L value
    assert not jnp.isnan(res_stab.L)
    assert res_stab.L > 0  # Stable


def test_10m_wind_and_2m_temp(bd_mo_w_th_s, use_jit):
    """Test the diagnostic 10m wind and 2m temperature outputs"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.01)

    # 10m wind should be positive but less than the reference height wind (due to log profile)
    assert 0 < res.m10 < 6.0

    # In unstable conditions with upward heat flux, 2m temperature should be between
    # surface temperature and reference height temperature
    assert min(res.th_s, 290.0) <= res.th2 <= max(res.th_s, 290.0)


def test_prescribe_th_s(bd_mo_th_s, use_jit):
    """Test the model with prescribed surface temperature"""
    res_stab: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=295.0, th_s=290.0)
    res_neut: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=290.0, th_s=290.0)
    res_unst: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=290.0, th_s=295.0)

    assert jnp.isclose(res_neut.w_th, 0)
    assert res_unst.w_th > 0
    assert res_stab.w_th < 0


def test_sukanta_matlab(use_jit):
    """Compare with Sukanta's Matlab results"""
    z = 12.9032 / 2
    mo = init_mo_sfc(
        z0m=0.1,
        z0h=0.1,
        sim_funcs=BusingerDyerSimFuncs(b=5.0, gamma=15.0),
        z=z,
        z_grad=z,
        prescribe="w_th_s",
    )

    res = mo(u_0=8, v_0=0, th_0=265, w_th_s=-0.08)
    assert jnp.isclose(res.u_st, 0.751988632621145)
    assert jnp.isclose(res.L, 3.589721161447895e02)
    assert jnp.isclose(res.du_dz, 0.317580713832262)
    assert jnp.isclose(res.dv_dz, 0.0)
    assert jnp.isclose(res.dth_dz, 0.044928462436926)

    res: MOResult = mo(u_0=8, v_0=0, th_0=265, w_th_s=+0.08)
    assert jnp.isclose(res.u_st, 0.778360675518376)
    assert jnp.isclose(res.L, -3.980792561582825e02)
    assert jnp.isclose(res.du_dz, 0.285643618090545)
    assert jnp.isclose(res.dv_dz, 0.0)
    assert jnp.isclose(res.dth_dz, -0.035721087489367)
    # assert jnp.isclose(res.m10, 8.852801010222086)  # sukanta doesn't reevaluate psi for 10m
