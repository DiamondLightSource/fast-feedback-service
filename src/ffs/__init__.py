from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ffs")
except PackageNotFoundError:
    pass
