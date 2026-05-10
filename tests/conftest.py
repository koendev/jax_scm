"""Pytest setup: expose multiple host CPU devices for pmap-based tests.

Must run before JAX initializes its XLA backend, which is why this lives at
module level in conftest.py — pytest imports this before any test module.
"""

import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=4",
)
