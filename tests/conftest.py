"""Conftest: stub out optional heavy dependencies before any fiona imports.

This allows ``fiona.utils`` (and its multiprocessing worker functions) to be
imported under their canonical module name ``fiona.utils``, which is required
for multiprocessing pickling to work correctly in tests.
"""
import sys
import types


def _stub_module(name: str):
    """Insert a trivial stub into sys.modules for *name* and all parents."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        if full not in sys.modules:
            sys.modules[full] = types.ModuleType(full)


# Stub heavy/optional packages that fiona.general (and friends) need so that
# importing fiona.utils through the normal package path doesn't crash.
for _pkg in (
    "numexpr",
    "finufft",
    "jax",
    "jax.numpy",
    "jaxlib",
):
    _stub_module(_pkg)
