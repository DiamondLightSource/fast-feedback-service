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

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

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


def _scm_version() -> str:
    """Resolve the version with setuptools_scm, the same provider the
    python build uses, so the docs carry the version the package and
    binaries are built with. Needs the git tags to be present.

    It runs as a command, as cmake and CI run it, because the command
    reads the configuration in pyproject.toml and get_version() does
    not; called directly, it would count commits differently."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "setuptools_scm"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "0.0.0"
    return result.stdout.strip()


release = _scm_version()
version = ".".join(release.split(".")[:2])

# General configuration

extensions = [
    "breathe",
    "exhale",
    "myst_parser",
    "sphinx_design",
]

# Allow the README (Markdown) to be pulled into the toctree.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Generate heading anchors so the README's in-page links resolve.
myst_heading_anchors = 3

# ":::" fences, which the generated tab sets below are written in, so
# they nest around code blocks without counting backticks.
myst_enable_extensions = ["colon_fence"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output

html_theme = "furo"
html_title = "fast-feedback-service documentation"

# Breathe configuration

# Breathe reads the XML from the Doxygen run that Exhale triggers below.
breathe_projects = {"ffs": "./doxyoutput/xml"}
breathe_default_project = "ffs"
# Breathe skips members of .c/.cc/.cpp files when resolving unqualified
# names, which leaves file-local functions with empty pages. Search all
# files instead.
breathe_implementation_filename_extensions: list[str] = []

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
        # Anonymous namespaces are file-private and Doxygen can only
        # name them with a generated "@<digits>" placeholder.
        EXCLUDE_SYMBOLS  = @*
        # CUDA qualifiers and the host/device shims expand to nothing so
        # the parser sees plain C++ signatures.
        MACRO_EXPANSION  = YES
        EXPAND_ONLY_PREDEF = YES
        PREDEFINED       = __cplusplus=202002L \\
                           __global__= __device__= __host__= \\
                           __forceinline__= __shared__= __constant__= \\
                           __restrict__= __noinline__= FFS_HD= \\
                           DEVICE_HOST=
        """
    ),
}

# The language Exhale/Breathe should assume for the generated tree.
primary_domain = "cpp"
highlight_language = "cpp"

# README link rewriting

# Repository the README's relative links resolve against on the docs site.
REPO_URL = "https://github.com/DiamondLightSource/fast-feedback-service"

# A reference link definition pointing at a page or file of these docs,
# e.g. "[building]: docs/building.md".
# The docs/ directory itself, "[`docs/`]: docs/", is a repository link.
_DOCS_LINK_DEFINITION = re.compile(
    r"^(\[[^\]]+\]):\s+docs/(\S+\.\w+)$", flags=re.MULTILINE
)

# A reference link definition with any other repo-relative target, e.g.
# "[`src/`]: src/". Absolute URLs and in-page anchors are left alone.
_RELATIVE_LINK_DEFINITION = re.compile(
    r"^(\[[^\]]+\]):\s+(?!https?://|#|docs/\S+\.\w+$)(\S+)$",
    flags=re.MULTILINE,
)


def _absolute_readme_links(
    app: object, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    """Point the README's repo-relative links at GitHub while it is
    included into the docs. On GitHub the same links stay relative and
    follow whichever branch or tag is being viewed; inside Sphinx they
    would resolve as missing documents instead. GitHub redirects a
    tree/ URL to blob/ for files, so one form covers both. Links to
    pages and files of these docs are the exception: they are made
    relative to docs/, where the README is included, so they stay on the
    site and images are copied into it."""
    if relative_path.name != "README.md":
        return
    content[0] = _RELATIVE_LINK_DEFINITION.sub(
        lambda m: f"{m[1]}: {REPO_URL}/tree/main/{m[2]}", content[0]
    )
    content[0] = _DOCS_LINK_DEFINITION.sub(r"\1: \2", content[0])


# The README's centred HTML title, e.g. '<h1 align="center">Title</h1>'
_HTML_TITLE = re.compile(r'^<h1 align="center">(.+?)</h1>$', flags=re.MULTILINE)


def _markdown_readme_title(
    app: object, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    """Turn the README's HTML title back into a Markdown heading while it
    is included into the docs. GitHub only centres a heading written as
    HTML, but Sphinx takes the page title from Markdown headings alone."""
    if relative_path.name != "README.md":
        return
    content[0] = _HTML_TITLE.sub(r"# \1", content[0], count=1)


# The README's banner as a <picture>, a dark source and a light fallback,
# both under docs/
_THEMED_PICTURE = re.compile(
    r"^<picture>\s*"
    r'<source media="\(prefers-color-scheme: dark\)" srcset="docs/(?P<dark>\S+)">\s*'
    r'<img alt="(?P<alt>[^"]*)" src="docs/(?P<light>\S+)">\s*'
    r"</picture>$",
    flags=re.MULTILINE,
)


def _themed_image(path: str, alt: str, scheme: str) -> str:
    return (
        f"```{{image}} {path}\n:alt: {alt}\n:class: only-{scheme}\n"
        ":align: center\n```\n"
    )


def _themed_readme_picture(
    app: object, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    """Turn the README's themed <picture> into one image per theme while it
    is included into the docs. GitHub picks the source from the reader's
    GitHub theme, but on the site the theme toggle only changes the page,
    and Sphinx does not copy images named in raw HTML. Furo shows only the
    image whose only-dark or only-light class matches its theme."""
    if relative_path.name != "README.md":
        return
    content[0] = _THEMED_PICTURE.sub(
        lambda m: (
            _themed_image(m["dark"], m["alt"], "dark")
            + "\n"
            + _themed_image(m["light"], m["alt"], "light")
        ),
        content[0],
    )


# Tabs written as plain Markdown

# A block of alternatives between "<!-- tabs: GROUP -->" and
# "<!-- end tabs -->", each starting with a bold label such as
# "**Diamond:**". GitHub hides the comments and shows the alternatives in
# turn; the docs build makes them tabs, synced across the group.
_TABS_BLOCK = re.compile(
    r"^<!-- tabs: (?P<group>[\w-]+) -->\n(?P<body>.*?)^<!-- end tabs -->$",
    flags=re.MULTILINE | re.DOTALL,
)
_TAB_LABEL = re.compile(r"^\*\*(?P<label>[^*\n]+):\*\*[ \t]*", flags=re.MULTILINE)


def _tab_set(block: re.Match[str]) -> str:
    """Rewrite one block of alternatives as a sphinx-design tab set, one
    tab per label. Text before the first label has no tab to go in, so
    it is an error rather than silently dropped."""
    body = block["body"]
    labels = list(_TAB_LABEL.finditer(body))
    if not labels or body[: labels[0].start()].strip():
        raise ValueError(
            f"tabs block '{block['group']}' must start with a **Label:** line"
        )
    lines = ["::::{tab-set}", f":sync-group: {block['group']}", ""]
    for label, following in zip(labels, labels[1:] + [None]):
        end = following.start() if following else len(body)
        lines += [
            f":::{{tab-item}} {label['label']}",
            f":sync: {label['label'].lower().replace(' ', '-')}",
            "",
            body[label.end() : end].strip(),
            ":::",
            "",
        ]
    lines.append("::::")
    return "\n".join(lines)


def _markdown_tabs(app: object, docname: str, source: list[str]) -> None:
    """Turn the tabs blocks in a page into tab sets before MyST parses it."""
    source[0] = _TABS_BLOCK.sub(_tab_set, source[0])


def setup(app):
    app.connect("include-read", _absolute_readme_links)
    app.connect("include-read", _markdown_readme_title)
    app.connect("include-read", _themed_readme_picture)
    app.connect("source-read", _markdown_tabs)
