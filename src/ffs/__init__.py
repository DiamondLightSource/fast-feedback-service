from __future__ import annotations

try:
    from ffs._version import __version__
except ImportError:
    # A source tree imported without a build has no generated _version.py
    __version__ = "0.0.0.dev0+unknown"
