# Configuration file for the Sphinx documentation builder.
#
# The documented code is C++ and CUDA, so Sphinx is used purely as the
# presentation layer over Doxygen. The pipeline is:
#
#     Doxygen (parse comments -> XML)
#       -> Breathe (expose the XML to Sphinx)
#       -> Exhale (auto-generate a page per class/file/namespace)
#       -> Sphinx (render HTML)
#
# Exhale runs Doxygen for us (exhaleExecutesDoxygen = True) with the
# Doxygen configuration inlined below, so there is no separate Doxyfile
# to keep in sync.

import textwrap

# Project information

project = "fast-feedback-service"
copyright = "Diamond Light Source"
# Contributors from the GitHub repository, ordered by contribution
# count. Update from
# `gh api repos/DiamondLightSource/fast-feedback-service/contributors`
# as needed.
author = (
    "Nicholas Devenish, Dimitri Vlachos, James Beilsten-Edmands, "
    "Graeme Winter, Yash Karan"
)


def _scm_version():
    """Resolve the version with setuptools_scm, the same provider the
    python build uses, so the docs carry the version the package and
    binaries are built with. Needs the git tags to be present."""
    try:
        from setuptools_scm import get_version

        return get_version(root="..", relative_to=__file__)
    except (ImportError, LookupError):
        return "0.0.0"


release = _scm_version()
version = ".".join(release.split(".")[:2])

# General configuration

extensions = [
    "breathe",
    "exhale",
    "myst_parser",
]

# Allow the README (Markdown) to be pulled into the toctree.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Generate heading anchors so the README's in-page links resolve.
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output

html_theme = "furo"
html_title = "fast-feedback-service documentation"

# Breathe configuration

# Breathe reads the XML from the Doxygen run that Exhale triggers below.
breathe_projects = {"ffs": "./doxyoutput/xml"}
breathe_default_project = "ffs"

# Exhale configuration

exhale_args = {
    # Where Exhale writes the generated .rst API tree (git-ignored).
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "API Reference",
    "doxygenStripFromPath": "..",
    # Let Exhale invoke Doxygen; the config below is fed to it on stdin.
    "createTreeView": True,
    "contentsDirectives": False,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": textwrap.dedent(
        """\
        INPUT            = ../include ../src ../integrator ../spotfinder \\
                           ../baseline ../h5read
        RECURSIVE        = YES
        FILE_PATTERNS    = *.hpp *.h *.cuh *.cc *.cpp *.cu
        # Tests are not API, and dx2 is a submodule with its own docs.
        EXCLUDE_PATTERNS = */tests/*
        # Doxygen does not know CUDA sources; parse them as C++.
        EXTENSION_MAPPING = cu=C++ cuh=C++
        # Document everything the parser sees.
        EXTRACT_ALL      = YES
        EXTRACT_STATIC   = YES
        HIDE_UNDOC_MEMBERS = NO
        # CUDA qualifiers and the host/device shim expand to nothing so
        # the parser sees plain C++ signatures.
        MACRO_EXPANSION  = YES
        EXPAND_ONLY_PREDEF = YES
        PREDEFINED       = __cplusplus=202002L \\
                           __global__= __device__= __host__= \\
                           __forceinline__= __shared__= __constant__= \\
                           __restrict__= __noinline__= FFS_HD=
        """
    ),
}

# The language Exhale/Breathe should assume for the generated tree.
primary_domain = "cpp"
highlight_language = "cpp"
