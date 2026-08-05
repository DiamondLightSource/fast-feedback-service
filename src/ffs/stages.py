"""
Command lines for the indexer and the integrator.

Both are configured the same way in every pipeline that runs them, so
their parameters and options live here rather than in an entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ffs.pipeline import append_optional

# Written by the indexer under fixed names, relative to the CWD
INDEXED_EXPERIMENTS = Path("indexed.expt")
INDEXED_REFLECTIONS = Path("indexed.refl")


class PipelineRequest(BaseModel):
    """Parameters for the pipeline."""

    # Inputs
    reflection: Path
    experiment: Path
    working_directory: Path

    max_cell: float

    dmin: Optional[float] = None
    dcid: Optional[int] = None

    # Indexer
    max_refine: Optional[int] = None
    macro_cycles: Optional[int] = None
    nthreads: Optional[int] = None

    # Integrator
    output: Path = Path("integrated.refl")
    algorithm: Optional[str] = None
    background: Optional[str] = None
    sigma_b: Optional[float] = None
    sigma_m: Optional[float] = None
    min_zeta: Optional[float] = None
    min_bbox_depth: Optional[int] = None
    threads: Optional[int] = None
    timeout: Optional[float] = None


def build_indexer_command(executable: Path, params: PipelineRequest) -> list[str]:
    """
    Assemble the baseline indexer command line.

    Args:
        executable: Path to the baseline_indexer binary
        params:     The validated request

    Returns:
        list[str]: The argv to hand to subprocess
    """
    command = [
        str(executable),
        "--refl",
        str(params.reflection),
        "--expt",
        str(params.experiment),
        "--max-cell",
        str(params.max_cell),
    ]
    append_optional(
        command,
        {
            "--dmin": params.dmin,
            "--max-refine": params.max_refine,
            "--macro-cycles": params.macro_cycles,
            "--nthreads": params.nthreads,
        },
    )
    return command


def build_integrator_command(
    executable: Path, params: PipelineRequest, output: Path
) -> list[str]:
    """
    Assemble the integrator command line.

    Reads the indexer's output directly: it predicts internally when
    the reflection flags say the table is not, and estimates the sigmas
    from the variance columns carried through from the strong spots.

    Args:
        executable: Path to the integrator binary
        params:     The validated request
        output:     Resolved path for the integrated reflections

    Returns:
        list[str]: The argv to hand to subprocess
    """
    command = [
        str(executable),
        "--reflection",
        str(INDEXED_REFLECTIONS),
        "--experiment",
        str(INDEXED_EXPERIMENTS),
        "--output",
        str(output),
    ]
    append_optional(
        command,
        {
            "--algorithm": params.algorithm,
            "--background": params.background,
            "--sigma_b": params.sigma_b,
            "--sigma_m": params.sigma_m,
            "--min_zeta": params.min_zeta,
            "--min_bbox_depth": params.min_bbox_depth,
            "--threads": params.threads,
            "--timeout": params.timeout,
        },
    )
    return command


def add_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the indexer and integrator options.

    Stages ahead of these add their own flags, prefixed where they
    collide.

    Args:
        parser: Parser to add the arguments to
    """
    parser.add_argument(
        "--dcid",
        type=int,
        help="Data collection ID, for logging",
    )
    parser.add_argument(
        "--max-refine",
        type=int,
        help="Candidate lattices to refine",
    )
    parser.add_argument(
        "--macro-cycles",
        type=int,
        help="Post-indexing macrocycles",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="Threads for the indexer FFT",
    )
    parser.add_argument(
        "--output",
        help="Integrated reflection table filename",
    )
    parser.add_argument(
        "--algorithm",
        help="Foreground algorithm: dials or ellipsoid",
    )
    parser.add_argument(
        "--background",
        help="Background model: constant, tukey or glm",
    )
    parser.add_argument(
        "--sigma-b",
        dest="sigma_b",
        type=float,
        help="σ_b in degrees",
    )
    parser.add_argument(
        "--sigma-m",
        dest="sigma_m",
        type=float,
        help="σ_m in degrees",
    )
    parser.add_argument(
        "--min-zeta",
        dest="min_zeta",
        type=float,
        help="Minimum zeta",
    )
    parser.add_argument(
        "--min-bbox-depth",
        dest="min_bbox_depth",
        type=int,
        help="Images a reflection must span to contribute to sigma estimation",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Integrator reader threads",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Seconds to wait for images",
    )
