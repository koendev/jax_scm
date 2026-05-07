# scm_ui — TODO / Known Issues

This document tracks known bugs and planned work for the UI package.
`result_viewer.py` / `cli.py view` is the only public-facing surface; `app.py` and `ic_bc_edit.py` are experimental.

---

## result_viewer.py

Note: viewer is SCM-output only. `mo_u_st` and `m` are always present in SCM datasets — no fallback needed.

### Bugs
- **`TsPlot` log scale removed**: `p.y_scale = bm.LogScale()` does not work on a live Bokeh figure.
  Removed the broken toggle. Proper fix needs figure rebuild or `CustomJS` — defer to new features.

### TODO (new features)
- Hover tooltip on `FieldPlot` — read (time, height, value) at cursor position.
- Vertical profile plot (value vs. z at a selected time) — key diagnostic for atmospheric science.
- `vars_2d` selector is a flat unsorted list of ~26 entries. Group by:
  prognostic (u, v, th, qv, qke) → diagnostic → MO surface (mo_*) → forcing (frc_*).
- No header showing simulation name, grid info (Nz, H), or time range from `ds.attrs`.

---

## app.py  (experimental, not public-facing)

### Bugs
- `from ic_bc_edit import MYNNEditor` — bare import, only works if cwd is `src/scm_ui/`.
  Fix: `from scm_ui.ic_bc_edit import MYNNEditor`.
- `Button.on_click` passes `n_clicks` to callback; `run(self)` accepts none → `TypeError` on click.
- `run()` is called synchronously on the Tornado IO loop — freezes the server during JAX compile + sim.
  Needs `ThreadPoolExecutor` / `run_in_executor`.
- "Model output" tab only contains the Run button; results are never displayed after a run.

### Hardcoded parameters (all need UI exposure)
- `MOSettings(z0h=0.1, z0m=0.1)`, `t_end_s=24*3600`, `th_ref=300.0`
- `dt_s_init=0.001`, `dt_s_max=1`, `cfl_max=0.05`
- Output path `"out_ui.nc"` in cwd — should be temp file or user-chosen.

### Missing
- No CLI entrypoint (`cli.py` only has `view`, no `run`/`edit` command).

---

## ic_bc_edit.py  (experimental, not public-facing)

### Bugs
- `Button.on_click` / `interp_callback(self)` same signature mismatch as `app.py`.
- `assert` used for input validation — silently skipped with `python -O`. Use `raise ValueError`.
- `_reset()` clears `pts` but not `line` datasource → stale interpolated profile persists visually.

### Physical / UX
- `qv` editor range `(-0.1, 0.1)` allows negative specific humidity. Range should start at 0.
- `f_c=1e-4` (Coriolis) hardcoded in `get_forcing()`, not exposed in UI.
- BC time axis is in hours, forcing uses seconds — easy source of confusion, add unit labels.