"""Low-level helpers for rendering figures and values used by the HTML reporter."""

import base64
import io
import pathlib

import jinja2
import matplotlib.pyplot as plt
import numpy as np

reporter_root = pathlib.Path(__file__).parent


def dict_to_css(d: dict) -> str:
    """Convert dict to CSS string."""
    return "; ".join([f"{k}: {v}" for k, v in d.items()])


def fig_to_png_base64(fig: plt.Figure, **save_kwargs) -> str:
    """Render a matplotlib figure to PNG and return the base64-encoded string.

    Parameters
    ----------
    fig : plt.Figure
        Figure to render.
    **save_kwargs
        Additional keyword arguments forwarded to ``fig.savefig``.

    Returns
    -------
    str
        UTF-8 base64 string of the PNG image, suitable for embedding in an
        HTML ``<img>`` ``src`` attribute.
    """
    fig_buffer = io.BytesIO()
    fig.savefig(fig_buffer, format="png", **save_kwargs)
    fig_buffer.seek(0)
    return base64.b64encode(fig_buffer.read()).decode("utf-8")


def fig_to_svg_base64(fig: plt.Figure, **save_kwargs) -> str:
    """Render a matplotlib figure to SVG and return the base64-encoded string.

    Parameters
    ----------
    fig : plt.Figure
        Figure to render.
    **save_kwargs
        Additional keyword arguments forwarded to ``fig.savefig``.

    Returns
    -------
    str
        UTF-8 base64 string of the SVG image, suitable for embedding in an
        HTML ``<img>`` ``src`` attribute.
    """
    fig_buffer = io.StringIO()
    fig.savefig(fig_buffer, format="svg", **save_kwargs)
    fig_svg_str = fig_buffer.getvalue()
    return base64.b64encode(fig_svg_str.encode("utf-8")).decode("utf-8")


def rad_to_deg_str(rad: float, pos: float = 0) -> str:
    """Takes ``rad`` in radian and returns value as formatted 2-decimal string in degree"""
    return f"{np.rad2deg(rad):.2f}°"


def float_to_str(f: float) -> str:
    """Returns float as string with two digits"""
    return f"{f:.2f}"


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(reporter_root / "templates"),
    autoescape=jinja2.select_autoescape(),
)
env.filters["rad2deg"] = np.rad2deg
env.filters["to_css"] = dict_to_css
