"""Setuptools options that have no declarative form in pyproject.toml.

The extension modules are built by cmake and collected as package data, so
setuptools cannot see that this project produces compiled output and would
otherwise call the distribution pure Python.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """A distribution carrying compiled modules whatever setuptools infers.

    Tags the wheel for the interpreter and platform it was built for, so a
    mismatched interpreter refuses it at install time instead of accepting
    it and failing on the first import, and places the modules in platlib
    rather than purelib.
    """

    def has_ext_modules(self) -> bool:
        return True


setup(
    distclass=BinaryDistribution,
    # Keep the staging tree out of the cmake build directory. It is what
    # the wheel is zipped from and setuptools never prunes it, so sharing
    # a directory leaves files from an older configuration to be shipped.
    options={"build": {"build_base": "build_python"}},
)
