from __future__ import annotations

import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.plotting as bp
import numpy as np
import xarray as xr

WIDTH = 1200


class DatasetWrapper:
    def __init__(self, ds: xr.Dataset):
        ds = ds.copy()
        ds["m"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
        ds["d"] = np.rad2deg(np.arctan2(-ds["u"], -ds["v"])) % 360  # meteorological [0, 360]

        self.ds = ds
        self.H = ds["zh"].values[-1]
        self.t_start, self.t_end = ds["time"].values[[0, -1]]

        self.vars_1d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 1 and not str(v).startswith("_")]
        self.vars_2d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 2]

    def get_ts_store(self) -> bm.ColumnDataSource:
        return bm.ColumnDataSource(data={v: self.ds[v].values for v in self.vars_1d + ["time"]})


class TsPlot:
    def __init__(self, dw: DatasetWrapper, var: str):
        p = bp.figure(height=300, width=WIDTH)
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (dw.t_start, dw.t_end)
        p.xaxis.axis_label = "time (h)"
        p.yaxis.axis_label = var

        line = p.line(x="time", y=var, source=dw.get_ts_store())

        def _sel_callback(attr, old, new):
            line.glyph.y = new
            p.yaxis.axis_label = new

        select = bm.Select(title="Select variable", options=dw.vars_1d, value=var)
        select.on_change("value", _sel_callback)

        self.p = p
        self.select = select

    def get_layout(self):
        return bl.column(self.select, self.p)


class FieldPlot:
    CMAP_SEQ = "Magma256"
    CMAP_DIV = "RdBu11"

    def __init__(self, dw: DatasetWrapper, var: str):
        p = bp.figure(height=500, width=WIDTH)
        color_mapper = bm.LinearColorMapper(palette=self.CMAP_SEQ, low=0, high=1)
        img = p.image(
            image=[],
            x=dw.t_start,
            y=0,
            dw=dw.t_end - dw.t_start,
            dh=dw.H,
            color_mapper=color_mapper,
            level="image",
        )
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (dw.t_start, dw.t_end)
        p.y_range.bounds = (0, dw.H)
        p.xaxis.axis_label = "time (h)"
        p.yaxis.axis_label = "height (m)"

        cbar = bm.ColorBar(color_mapper=color_mapper)
        p.add_layout(cbar, "right")

        vrange = bm.RangeSlider(start=0, end=1, value=(0, 1), step=0.01, title="Color range")
        vrange.on_change("value", self._vrange_callback)

        select = bm.Select(title="Select variable", options=dw.vars_2d, value=var)
        select.on_change("value", self._sel_callback)

        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", self._log_callback)

        div_toggle = bm.Switch(label="Divergent colors")
        div_toggle.on_change("active", self._div_color_callback)

        self.p = p
        self.select = select
        self.log_toggle = log_toggle
        self.vrange = vrange
        self.div_toggle = div_toggle
        self._img = img
        self._color_mapper = color_mapper
        self._vrange_updating = False

        self.var = var
        self.dw = dw
        self._update_data(var=var, use_log=False)

    def _update_vmin_vmax(self, vmin: float, vmax: float):
        self._color_mapper.low = vmin
        self._color_mapper.high = vmax

        self.vrange.start = vmin
        self.vrange.end = vmax
        self.vrange.value = (vmin, vmax)
        self.vrange.step = max((vmax - vmin) / 250, 1e-10)

    def _update_data(self, var: str | None, use_log: bool):
        if var is None:
            var = self.var
        else:
            self.var = var

        data = self.dw.ds[var].values.T
        data = np.log10(data) if use_log else data
        data = np.where(np.isfinite(data), data, np.nan)

        self._img.data_source.data["image"] = [data]
        self._update_vmin_vmax(vmin=np.nanmin(data), vmax=np.nanmax(data))

    def _sel_callback(self, attr, old, new_var):
        self._update_data(var=new_var, use_log=self.log_toggle.active)
        if self.div_toggle.active:
            self._div_color_callback(None, None, True)

    def _vrange_callback(self, attr, old, new):
        if self._vrange_updating:
            return

        new_min, new_max = new
        if self.div_toggle.active:
            old_min, old_max = old
            if old_min != new_min:
                new_max = new_min * -1
            elif old_max != new_max:
                new_min = new_max * -1

            self._vrange_updating = True
            self.vrange.value = (new_min, new_max)
            self._vrange_updating = False

        self._color_mapper.low = new_min
        self._color_mapper.high = new_max

    def _log_callback(self, attr, old, use_log):
        self.div_toggle.disabled = use_log
        if self.div_toggle.active and use_log:
            self.div_toggle.active = False
        self._update_data(var=None, use_log=use_log)

    def _div_color_callback(self, attr, old, use_div):
        vmin = self.dw.ds[self.var].values.min()
        vmax = self.dw.ds[self.var].values.max()

        if use_div:
            self.log_toggle.active = False
            self._color_mapper.palette = self.CMAP_DIV
            vsym = max(abs(vmin), abs(vmax))
            self._update_vmin_vmax(vmin=-vsym, vmax=vsym)
        else:
            self._color_mapper.palette = self.CMAP_SEQ
            self._update_vmin_vmax(vmin=vmin, vmax=vmax)

    def get_layout(self):
        return bl.column(
            bl.row(
                self.select,
                self.vrange,
                bl.column(
                    self.log_toggle,
                    self.div_toggle,
                ),
            ),
            self.p,
        )


class ResultViewer:
    def __init__(self, ds: xr.Dataset):
        self.dw = DatasetWrapper(ds)

    def get_layout(self):
        return bl.column(
            FieldPlot(dw=self.dw, var="m").get_layout(),
            TsPlot(dw=self.dw, var="mo_u_st").get_layout(),
        )
