"""Custom YAML serializers with ``pathlib.Path`` and tuple round-trip support."""

from __future__ import annotations

import pathlib
from typing import Dict

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def path_representer(dumper: yaml.Dumper, path: pathlib.Path):
    """Represent ``pathlib.Path`` as string in yaml."""
    return dumper.represent_scalar(tag="!path", value=str(path))


def path_constructor(loader: yaml.loader, node) -> pathlib.Path:
    """Convert string back to ``pathlib.Path`` object."""
    return pathlib.Path(loader.construct_scalar(node))


def tuple_representer(dumper: yaml.Dumper, t: tuple):
    """Convert tuple to yaml list. Attention! Deserialisation will be list not tuple! Pydantic will fix that."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", t, flow_style=True)


# Register path representer and constructor for Posix and Windows to Dumper
yaml.add_representer(pathlib.Path, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.PosixPath, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.WindowsPath, path_representer, Dumper=Dumper)
yaml.add_constructor("!path", path_constructor, Loader=Loader)

# Register representer and constructor to convert tuple between Python and yaml.
yaml.add_representer(tuple, tuple_representer, Dumper=Dumper)


def yaml_to_dict(yaml_str: str) -> Dict:
    """Parse a YAML string into a plain Python dictionary.

    Parameters
    ----------
    yaml_str : str
        YAML-formatted text.  An empty string returns an empty dict.

    Returns
    -------
    dict
        Parsed key-value mapping; nested structures become nested dicts.
    """
    if yaml_str == "":
        return {}
    return yaml.load(yaml_str, Loader=Loader)


def dict_to_yaml(d: Dict) -> str:
    """Serialise a Python dictionary to a YAML string.

    Parameters
    ----------
    d : dict
        Mapping to serialise.  ``pathlib.Path`` values are written with the
        custom ``!path`` tag; tuples are written as inline YAML sequences.

    Returns
    -------
    str
        YAML-formatted text representation of *d*.
    """
    return yaml.dump(d, Dumper=Dumper)
