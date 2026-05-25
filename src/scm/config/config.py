"""Pydantic configuration models and YAML namelist loader for JAX-SCM."""

from __future__ import annotations

import pathlib
import warnings
from enum import StrEnum
from typing import Self

import pydantic

from scm.config.yaml import yaml_to_dict


class TimeIntMethod(StrEnum):
    """Time integration method."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class AdaptiveTimestepConfig(pydantic.BaseModel):
    """CFL-based adaptive time-step settings for the explicit AB2 scheme.

    The time step is computed as ``dt = cfl_max * dz² / K_max`` and capped at
    ``dt_s_max``.  ``cfl_max`` must not exceed 0.5 (AB2 diffusion stability
    limit).  Ignored when ``time_int == "implicit"``.
    """

    cfl_max: float = 0.5  # Max CFL number for diffusion
    dt_s_max: float = 10.0  # Maximum time step, seconds

    @pydantic.field_validator("cfl_max")
    @classmethod
    def cfl_max_stable_for_ab2(cls, v: float) -> float:
        if v > 0.5:
            raise ValueError(
                f"cfl_max={v} exceeds 0.5, which is outside the AB2 stability region "
                "for diffusion.  Use cfl_max ≤ 0.5 or switch to implicit time stepping."
            )
        return v


class LogLevel(StrEnum):
    """Verbosity levels for simulation progress output."""

    SILENT = "silent"  # nothing is printed
    BEGIN_END = "begin_end"  # only "begin" and "complete" messages
    STEPS = "steps"  # boundary + per-outer-step progress with ETA


class LogConfig(pydantic.BaseModel):
    """Logging configuration for ``simulate``.

    Controls verbosity and how frequently progress lines are printed when
    ``level == LogLevel.STEPS``.
    """

    level: LogLevel = LogLevel.STEPS
    log_every_n: int = 1  # Only log every n outer steps (ignored if level is not STEPS)


class Namelist(pydantic.BaseModel):
    """Main configuration

    Always set default values and add comment to each field for documentation.
    """

    # Implicit or explicit time integration.
    time_int: TimeIntMethod = TimeIntMethod.EXPLICIT

    # If explicit, use adaptive time stepping or constant time step.
    adaptive_timestep: AdaptiveTimestepConfig | None = None

    # Integration time step, seconds
    dt_s: float = 0.01

    # Output time step, seconds
    dt_s_out: float = 5 * 60.0  # 5 mins

    logging: LogConfig = LogConfig()

    mo_n_iter: int = 10  # Number of iterations for MO solver. Increase for more accuracy, but also more runtime.

    @pydantic.model_validator(mode="after")
    def implicit_no_adaptive(self) -> Self:
        """Enforce that adaptive time stepping is disabled for implicit schemes.

        Returns
        -------
        Self
            The validated model instance, with ``adaptive_timestep`` set to
            ``None`` and a warning emitted when implicit integration was
            requested alongside an ``AdaptiveTimestepConfig``.
        """
        if self.time_int == TimeIntMethod.IMPLICIT and self.adaptive_timestep is not None:
            self.adaptive_timestep = None
            warnings.warn("Implicit time stepping set, ignoring adaptive_timestep config.")
        return self

    @property
    def is_implicit(self) -> bool:
        return self.time_int == TimeIntMethod.IMPLICIT


def load_namelist(f: str | pathlib.Path) -> Namelist:
    """Parse a YAML namelist file into a validated :class:`Namelist` instance.

    Parameters
    ----------
    f : str or pathlib.Path
        Path to a YAML file whose top-level keys correspond to ``Namelist``
        fields.

    Returns
    -------
    Namelist
        Fully validated configuration object.
    """
    f = pathlib.Path(f)
    f_dict = yaml_to_dict(f.read_text())
    return Namelist(**f_dict)
