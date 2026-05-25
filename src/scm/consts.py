"""Physical and numerical constants used throughout JAX-SCM."""

#: Gravitational acceleration (m s⁻²).
g = 9.81  # m/s^2
#: von Kármán constant (dimensionless).
kappa = 0.4  # von-Karman constant
#: Reference air density (kg m⁻³).
rho_0 = 1.225  # kg/m^3
#: Specific heat of dry air at constant pressure (J kg⁻¹ K⁻¹).
cp = 1005.0  # J/(kg K)
#: Ratio of specific heats for dry air (dimensionless).
gamma = 1.4
#: Specific gas constant for dry air (J kg⁻¹ K⁻¹).
Rd = 287.0  # J/(kg K)

#: Latent heat of vaporization of water (J kg⁻¹).
L_v = 2257e3  # J/kg, latent heat of vaporization of water

# Physical state floors (applied in clip_state after every step)
#: Minimum value of q² = 2·TKE (m² s⁻²); prevents ``sqrt(0)`` in the closure.
qke_min = 1e-10  # minimum q^2=2*TKE to avoid sqrt(0) in closure

# Numerical guards for differentiability
#: Floor applied to the argument of fractional-power expressions in ``safe_root`` to keep gradients finite.
smooth_eps = 1e-10  # floor for x^(frac<1) expressions used in safe_root to keep gradients finite
#: Minimum eddy diffusivity used as the CFL denominator (m² s⁻¹).
K_min = 1e-6  # minimum eddy diffusivity for CFL denominator
#: Minimum turbulent length scale used in the dissipation denominator (m).
L_min = 1e-3  # minimum length scale in dissipation denominator
