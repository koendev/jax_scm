from __future__ import annotations
import numpy as np
import xarray as xr
import bokeh.plotting as bp
import bokeh.models as bm
import bokeh.layouts as bl

ds = xr.open_dataset("../out.nc")
H = ds["zh"].values[-1]  # total depth
t_start, t_end = ds["time"].values[[0, -1]]

vars_1d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 1] + ["time"]
store_1d = bm.ColumnDataSource(data={v: ds[v].values for v in vars_1d})

vars_2d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 2]


class TsPlot:
    def __init__(self, var: str, linked: TsPlot | None = None):
        # Create figure with shared x_range if linked, otherwise default to time range
        fig_kwargs = {}
        if linked:
            fig_kwargs["x_range"] = linked.p.x_range

        p = bp.figure(height=300, width=700, **fig_kwargs)
        if not linked:
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.x_range.bounds = (t_start, t_end)

        # Plot line
        line = p.line(x="time", y=var, source=store_1d)

        # Selector
        def _sel_callback(attr, old, new):
            line.glyph.y = new

        select = bm.Select(title="Select variable", options=vars_1d[:-1], value=var)
        select.on_change("value", _sel_callback)

        # Log switch
        def _log_callback(attr, old, new):
            if new:
                p.y_scale = bm.LogScale()
            else:
                p.y_scale = bm.LinearScale()

        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", _log_callback)

        self.p = p
        self.select = select
        self.log_toggle = log_toggle

    def get_layout(self):
        return bl.column(
            bl.row(self.select, self.log_toggle),
            self.p,
        )


class FieldPlot:
    def __init__(self, var: str):
        p = bp.figure(height=500, width=700)
        color_mapper = bm.LinearColorMapper(palette="Sunset11", low=0, high=1)  # dummy values
        img = p.image(
            image=[],  # will be populated by `_update_data` call
            x=t_start,
            y=0,
            dw=t_end - t_start,
            dh=H,
            color_mapper=color_mapper,
            level="image",
        )
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (t_start, t_end)
        p.y_range.bounds = (0, H)

        # Add color bar
        cbar = bm.ColorBar(color_mapper=color_mapper)
        p.add_layout(cbar, "right")

        # Color bar ranges selector
        def _vrange_callback(attr, old, new):
            """Set colorbar to range from range selector"""
            color_mapper.low = new[0]
            color_mapper.high = new[1]

        vrange = bm.RangeSlider(start=0, end=1, value=(0, 1), step=10)  # dummy values
        vrange.on_change("value", _vrange_callback)

        def _update_data(var: str | None, use_log: bool):
            """Load image data, apply log, and update image, color mapper, and color range selector."""
            if var is None:
                var = self.var  # reuse currently selected variable
            else:
                self.var = var  # update currently selected variable

            # Load data
            data = ds[var].values.T
            data = np.log10(data) if use_log else data
            data = np.where(np.isfinite(data), data, np.nan)  # log can introduce -inf, so force to NaN

            # Update image
            img.data_source.data["image"] = [data]
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)

            # Update colormapper
            color_mapper.low = vmin
            color_mapper.high = vmax

            # Update vrange slider
            vrange.start = vmin
            vrange.end = vmax
            vrange.value = (vmin, vmax)
            vrange.step = (vmax - vmin) / 250

        # Variable selector
        def _sel_callback(attr, old, new_var):
            _update_data(var=new_var, use_log=self.log_toggle.active)

        select = bm.Select(title="Select variable", options=vars_2d, value=var)
        select.on_change("value", _sel_callback)

        # Log scale toggle
        def _log_callback(attr, old, use_log):
            """Keep current variable, but update log scale"""
            _update_data(var=None, use_log=use_log)

        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", _log_callback)

        # Now set data
        self.var = var
        _update_data(var=var, use_log=False)

        self.p = p
        self.select = select
        self.log_toggle = log_toggle
        self.vrange = vrange

    def get_layout(self):
        return bl.column(
            bl.row(self.select, self.log_toggle, self.vrange),
            self.p,
        )


bp.curdoc().add_root(
    bl.column(
        FieldPlot("ct2").get_layout(),
        TsPlot("mo_u_st").get_layout(),
    )
)
