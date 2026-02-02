from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image


def get_ref_ax(
    img_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    trim: Tuple[int, int, int, int] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # Load image and trim if needed
    img = Image.open(img_path)
    if trim is not None:
        w, h = img.size
        left, bottom, right, top = trim
        img = img.crop(
            (
                0 + left,
                0 + top,
                w - right,
                h - bottom,
            )
        )  # (left, top) to  (right, bottom)

    fig, ax = plt.subplots()

    # format: [xmin, xmax, ymin, ymax]
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto")

    return fig, ax


if __name__ == "__main__":
    res = xr.open_dataset("out.nc")
    t_min = res["time"] * 60  # hours to minutes

    # fig, ax = get_ref_ax(
    #     "ref_cuxart06/fig02_ust.png",
    #     (0, 540),
    #     (0.2, 0.5),
    #     trim=(109, 70, 20, 15),
    # )
    # ax.plot(t_min, res["mo_u_st"], lw=2, c="red")
    # ax.set_xlabel("Time, mins")
    # ax.set_ylabel("$u_*$, m/s")
    # fig.show()

    m = np.sqrt(res["u"] ** 2 + res["v"] ** 2)
    fig, ax = get_ref_ax(
        "ref_cuxart06/fig03_m.png",
        (0, 11),
        (0, 400),
        trim=(112, 66, 13, 15),
    )
    ax.plot(m.isel(time=-1), res["z"], lw=2, c="red")
    ax.set_xlabel("Wind speed, m/s")
    ax.set_ylabel("Height, m")

    fig.show()
