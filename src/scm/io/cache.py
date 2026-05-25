"""Disk-backed cache for xarray Datasets to avoid redundant network downloads."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import pickle
from functools import wraps
from typing import Callable

import xarray as xr

logger = logging.getLogger("scm.io.cache")


class XRCache:
    """Disk-backed cache that persists xarray Datasets as NetCDF files.

    Results are keyed by an MD5 hash of the wrapped function's call arguments.
    Cache files are stored as ``<cache_dir>/<fn_name>_<hash>.nc``.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Directory in which cached NetCDF files are stored; created if absent.
    disable : bool, optional
        When ``True``, the cache is bypassed and the wrapped function is always
        called directly.  Defaults to ``False``.
    """

    def __init__(self, cache_dir: str | pathlib.Path, disable: bool = False):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disable = disable

    @staticmethod
    def get_hash(fn: Callable, *args, **kwargs) -> str:
        """Compute an MD5 hash from the positional and keyword arguments of a call.

        Parameters
        ----------
        fn : Callable
            The function being called (currently unused in the hash; reserved for
            future function-code hashing).
        *args
            Positional arguments passed to ``fn``.
        **kwargs
            Keyword arguments passed to ``fn``.

        Returns
        -------
        str
            Hexadecimal MD5 digest string uniquely identifying the argument set.
        """

        # Hash func code
        # This is based on https://github.com/joblib/joblib/blob/main/joblib/memory.py#L655
        # fn_code_h = hash(getattr(fn, "__code__", None))
        # fn_hash = (id(fn), hash(fn), fn_code_h)

        hasher = hashlib.md5()
        hasher.update(pickle.dumps((args, frozenset(kwargs.items()))))  # hash arguments
        # hasher.update(pickle.dumps(fn_hash))  # hash function code
        return hasher.hexdigest()

    def cache(self, fn: Callable[..., xr.Dataset]) -> Callable[..., xr.Dataset]:
        """Wrap a function so its xarray Dataset result is cached to disk.

        On a cache hit the Dataset is loaded from the existing NetCDF file; on a
        miss the function is called, the result is written to disk, and then
        returned.

        Parameters
        ----------
        fn : Callable[..., xr.Dataset]
            Function whose return value should be cached.

        Returns
        -------
        Callable[..., xr.Dataset]
            Wrapped function with identical signature that transparently reads
            from or writes to the cache.
        """

        @wraps(fn)
        def wrapper(*args, **kwargs) -> xr.Dataset:
            if self.disable:
                logger.debug("Cache disabled. Computing result without caching.")
                return fn(*args, **kwargs)

            # Compute hash from arguments and use it as cache file path
            input_hash = self.get_hash(fn, *args, **kwargs)
            cache_file = self.cache_dir / f"{fn.__name__}_{input_hash}.nc"

            if cache_file.exists():
                logger.debug(f"Cache hit: {cache_file}")
                ds = xr.open_dataset(cache_file)
            else:
                logger.debug(f"Cache miss: {cache_file}. Computing and caching result.")
                ds = fn(*args, **kwargs)
                ds = ds.load()  # load here so data is available when returned
                ds.to_netcdf(cache_file)
            return ds

        return wrapper
