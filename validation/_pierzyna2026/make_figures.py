from typing import Callable
import pathlib

import jax
import dataclasses
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from scm.examples import get_andren1994, get_wangara_day33, get_gabls1
from scm.interfaces import Simulation

sns.set_palette("colorblind")
plt.rcParams.update(
    {
        # "font.family": "serif",
        "font.size": 8,
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.dpi": 300,
        # "figure.labelsize": 8,
        "lines.linewidth": 1.0,
        "hatch.linewidth": 0.5,
    }
)


COLORS = {
    "u": "C0",
    "v": "C1",
    "th": "C2",
    "qv": "C3",
    "qke": "C4",
}

LABELS_PRETTY = {
    "u": "$U$",
    "v": "$V$",
    "th": r"$\Theta$",
    "qv": r"$Q_v$",
    "qke": r"$q^2$",
    "th_s": r"$\theta_s$",
    "w_th": r"$\langle w \theta \rangle$",
    "w_qv": r"$\langle w {q_v} \rangle$",
}
UNITS = {
    "u": "m s$^{-1}$",
    "v": "m s$^{-1}$",
    "th": "K",
    "qv": "g/kg",
    "qke": "m$^2$ s$^{-2}$",
    "th_s": "K",
    "w_th": "K m s$^{-1}$",
    "w_qv": "g/kg m s$^{-1}$",
}

FIG_ROOT = pathlib.Path("figures")


@dataclasses.dataclass
class SimDrawSpec:
    sim: Simulation
    short_name: str
    time_formatter: Callable[[float], str]
    time_label: str = "Time, s"
    time_n_ticks: int = 5


# Global simulation objects for plotting
sim_a94 = get_andren1994()
sim_gab1 = get_gabls1()
sim_wg33 = get_wangara_day33()
sims = [
    SimDrawSpec(
        sim=sim_a94,
        short_name="A94",
        time_formatter=lambda t: f"{t * sim_a94.forcing.f_c:.0f}",
        time_n_ticks=6,
        time_label="Time, f$^{-1}$",
    ),
    SimDrawSpec(
        sim=sim_gab1,
        short_name="GAB1",
        time_formatter=lambda t: f"{t/3600:.0f}",
        time_label="Time, h",
        time_n_ticks=10,
    ),
    SimDrawSpec(
        sim=sim_wg33,
        short_name="WG33",
        time_formatter=lambda t: f"{t/3600:02.0f}",
        time_label="Time, LST",
        time_n_ticks=8,
    ),
]


def _add_is_const(v: jnp.ndarray, ax: plt.Axes, x: float = 0.95, y: float = 0.95, color: str = "grey") -> None:
    """Add 'constant' label if plotted variable is constant"""
    if v.mean() == 0:
        label = "zero"
    elif jnp.abs(v.std() / v.mean()) < 1e-5:
        label = "constant"
    else:
        return

    if x == 0.5:
        ha = "center"
    elif x < 0.5:
        ha = "left"
    else:
        ha = "right"

    ax.text(x, y, label, transform=ax.transAxes, ha=ha, va="top", fontsize=6, color=color)


def plot_ic(sds: SimDrawSpec, fig: plt.Figure, gs: plt.SubplotSpec) -> None:
    """Plot initial conditions."""
    ic_gs = gs.subgridspec(nrows=1, ncols=4)
    ax_uv = fig.add_subplot(ic_gs[0, 0])
    ax_th = fig.add_subplot(ic_gs[0, 1], sharey=ax_uv)
    ax_qv = fig.add_subplot(ic_gs[0, 2], sharey=ax_uv)
    ax_qke = fig.add_subplot(ic_gs[0, 3], sharey=ax_uv)

    sim = sds.sim
    ax_uv.plot(sim.init.u, sim.grid.z, label="u", color=COLORS["u"])
    ax_uv.plot(sim.init.v, sim.grid.z, label="v", color=COLORS["v"])
    _add_is_const(v=sim.init.u, ax=ax_uv, x=0.5)
    ax_uv.legend()
    ax_uv.set_xlabel(f"Wind, {UNITS['u']}")

    ax_th.plot(sim.init.th, sim.grid.z, label="th", color=COLORS["th"])
    _add_is_const(v=sim.init.th, ax=ax_th, x=0.5)
    ax_th.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['th']}")

    ax_qv.plot(sim.init.qv * 100, sim.grid.z, label="qv", color=COLORS["qv"])
    _add_is_const(v=sim.init.qv, ax=ax_qv, x=0.5)
    ax_qv.set_xlim(0, None)
    ax_qv.set_xlabel(f"{LABELS_PRETTY['qv']}, {UNITS['qv']}")

    ax_qke.plot(sim.init.qke, sim.grid.z, label="qke", color=COLORS["qke"])
    ax_qke.set_xlim(0, None)
    ax_qke.set_xlabel(f"{LABELS_PRETTY['qke']}, {UNITS['qke']}")

    ax_uv.set_ylabel("Height, m")
    ax_uv.set_ylim(0, sim.grid.H)

    for ax in (ax_th, ax_qv, ax_qke):
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def plot_bc(sds: SimDrawSpec, fig: plt.Figure, gs: plt.SubplotSpec) -> None:
    """Plot boundary conditions (i.e. time-varying forcing)."""
    row_gs = gs.subgridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1, 3),
        height_ratios=(1, 1),
    )
    ax_ug = fig.add_subplot(row_gs[:, 0])
    ax_heat = fig.add_subplot(row_gs[0, 1])
    ax_w_qv = fig.add_subplot(row_gs[1, 1], sharex=ax_heat)

    sim = sds.sim
    t = jnp.linspace(sim.t_start_s, sim.t_end_s)
    t_ticks = jnp.linspace(sim.t_start_s, sim.t_end_s, sds.time_n_ticks)
    t_ticks_ug = jnp.linspace(sim.t_start_s, sim.t_end_s, 3)

    # Geostrophic forcing
    ug = jax.vmap(sim.forcing.u_geo)(t)
    pc = ax_ug.pcolormesh(t, sim.grid.z, ug.T, shading="auto", cmap="Blues", rasterized=True)
    _add_is_const(v=ug, ax=ax_ug, x=0.5, color="white")
    ax_ug.set_xticks(t_ticks_ug)
    ax_ug.set_xticklabels([sds.time_formatter(tick) for tick in t_ticks_ug])
    ax_ug.set_xlabel(sds.time_label)
    ax_ug.set_ylabel("Height, m")
    ax_ug.set_ylim(0, sim.grid.H)
    fig.colorbar(pc, ax=ax_ug, label="$U_g$, m s$^{-1}$", pad=0.01)

    # Heat forcing
    if sim.forcing.w_th_s is None:
        # Surface temperature forcing
        heat = jax.vmap(sim.forcing.th_s)(t)
        label = LABELS_PRETTY["th_s"]
        unit = UNITS["th_s"]
    else:
        # Sensible heat flux forcing
        heat = jax.vmap(sim.forcing.w_th_s)(t)
        label = LABELS_PRETTY["w_th"]
        unit = UNITS["w_th"]

    ax_heat.plot(t, heat, color=COLORS["th"])
    ax_heat.set_ylabel(f"{label},\n{unit}")
    ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_heat.margins(x=0)
    _add_is_const(v=heat, ax=ax_heat)

    # Moisture forcing
    w_qv = jax.vmap(sim.forcing.w_qv_s)(t) * 1e3
    ax_w_qv.plot(t, w_qv, label="w_qv", color=COLORS["qv"])
    _add_is_const(v=w_qv, ax=ax_w_qv)

    ax_w_qv.margins(x=0)
    ax_w_qv.set_xlabel(sds.time_label)
    ax_w_qv.set_xticks(t_ticks)
    ax_w_qv.set_xticklabels([sds.time_formatter(tick) for tick in t_ticks])

    ax_w_qv.set_ylabel(f"{LABELS_PRETTY['w_qv']},\n{UNITS['w_qv']}")


def plot_ic_bc(sds: SimDrawSpec) -> plt.Figure:
    """Plot initial conditions and boundary conditions for a given simulation."""
    fig = plt.figure(constrained_layout=True, figsize=(4, 3))
    outer_gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(1, 1))

    plot_ic(sds, fig, outer_gs[0])
    plot_bc(sds, fig, outer_gs[1])

    return fig


if __name__ == "__main__":
    for sim in sims:
        fig_ic_bc = plot_ic_bc(sim)
        fig_ic_bc.savefig(FIG_ROOT / f"ic_bc_{sim.short_name}.pdf")
