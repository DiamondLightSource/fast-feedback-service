#!/usr/bin/env python3
"""
Draw the README banner from the spots the GPU spotfinder finds.

Renders the art in banner.cc as a detector frame, one Gaussian spot per
glyph on a Poisson background, and hands it to the spotfinder through
its shared-memory reader. Each centroid it reports becomes a circle,
sized and coloured by the counts around it; the background dots are too
faint to make a spot, so they only show in the frame behind.

Needs a GPU and a built spotfinder, so the SVG is committed and this is
only rerun when the art or a colour map changes.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import bitshuffle
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where the art and the colour maps are defined
BANNER_SOURCE = REPO_ROOT / "src" / "ffs" / "banner.cc"

# Where the README and the docs pick the picture up
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images" / "banner.svg"

# The spotfinder a plain ./build.sh produces
DEFAULT_SPOTFINDER = REPO_ROOT / "build" / "bin" / "spotfinder"

# Pixels per character of art. A terminal cell is about twice as tall as it is
# wide and the art puts spots in every other column, so this keeps the lattice
# square at 16 px in both directions.
CELL_WIDTH_PX = 8
CELL_HEIGHT_PX = 16

# Empty detector around the art, in pixels
MARGIN_PX = 24

# Peak height above background, in counts, for each glyph. The three letter
# glyphs sit well above the dispersion threshold. The background glyph is low
# enough that no more of it passes than the minimum spot size discards. At 8
# counts one background spot was found with seed 3, so it sits at 5.
GLYPH_PEAK_COUNTS = {"●": 400.0, "•": 150.0, "∙": 60.0, "·": 5.0}

# The glyph that stands for background rather than a spot in the lettering
BACKGROUND_GLYPH = "·"

# Spot width, in pixels. Narrow enough that neighbouring spots 16 px apart
# never touch, wide enough to clear the spotfinder's minimum spot size.
SPOT_SIGMA_PX = 1.5

# Mean background, in counts per pixel, as on a photon-counting detector
BACKGROUND_COUNTS = 1.0

# Half-width of the box summed to measure a spot's intensity, in pixels
INTENSITY_BOX_HALF_PX = 3

# Seed for the Poisson noise, so a regeneration with unchanged inputs gives an
# identical picture
DEFAULT_SEED = 3

# Count ceiling written into the header. Far above any pixel here, so nothing
# is treated as overloaded.
TRUSTED_COUNT_CUTOFF = 60000

# Geometry written into the header. The spotfinder needs plausible values but
# the picture does not depend on them.
WAVELENGTH_A = 0.976
DETECTOR_DISTANCE_MM = 200.0
PIXEL_SIZE_M = 7.5e-5

# Number of colour steps between the faintest and the brightest found spot
COLOUR_STEPS = 8

# Spot radius in pixels at the faintest and the brightest found spot
SPOT_RADIUS_PX = (1.3, 2.9)

# Opacity of the brightest pixel of the detector frame behind the spots
FRAME_MAX_OPACITY = 0.55

# Scale of the rendered SVG relative to the detector, so it reads at README
# width without being resampled
DISPLAY_SCALE = 1.5


@dataclass(frozen=True)
class Palette:
    """One colour map from banner.cc, each ramp brightest spot first."""

    name: str
    dark: tuple[str, ...]
    light: tuple[str, ...]


@dataclass(frozen=True)
class Spot:
    """A spot the spotfinder reported, in detector pixels."""

    x: float
    y: float
    intensity: float


def read_banner_art(source: str) -> list[str]:
    """Return the lines of the spot art from the text of banner.cc."""
    match = re.search(r'banner_art = R"\(\n(.*?)\)";', source, flags=re.DOTALL)
    if match is None:
        raise ValueError("no banner_art raw string in banner.cc")
    return match[1].rstrip("\n").split("\n")


def read_palettes(source: str) -> list[Palette]:
    """Return the colour maps from the bike_shed table in banner.cc.

    Each entry opens with a "// Name: ..." comment and lists its dark ramp's
    four 24-bit colours, its 16-colour stand-ins, then the same for its light
    ramp. Only the 24-bit colours are used here.
    """
    match = re.search(r"bike_shed\{\{(.*?)\n\}\};", source, flags=re.DOTALL)
    if match is None:
        raise ValueError("no bike_shed table in banner.cc")
    entries = re.split(r"^\s*// ", match[1], flags=re.MULTILINE)[1:]
    palettes = []
    for entry in entries:
        name = entry.split(":", 1)[0].strip()
        colours = [
            f"#{hex_value}"
            for hex_value in re.findall(r"rgb\(0x([0-9A-Fa-f]{6})\)", entry)
        ]
        if len(colours) != 8:
            raise ValueError(f"palette {name!r} has {len(colours)} colours, expected 8")
        palettes.append(Palette(name, tuple(colours[:4]), tuple(colours[4:])))
    return palettes


def art_spots(art: list[str]) -> list[tuple[float, float, str]]:
    """Return each glyph of the art as (x, y, glyph), in detector pixels."""
    spots = []
    for row, line in enumerate(art):
        for column, glyph in enumerate(line):
            if glyph in GLYPH_PEAK_COUNTS:
                x = MARGIN_PX + (column + 0.5) * CELL_WIDTH_PX
                y = MARGIN_PX + (row + 0.5) * CELL_HEIGHT_PX
                spots.append((x, y, glyph))
    return spots


def render_frame(
    shape: tuple[int, int], spots: list[tuple[float, float, str]], seed: int
) -> np.ndarray:
    """Render a detector frame: Gaussian spots on background, with Poisson noise.

    Coordinates follow the spotfinder's convention, in which pixel i spans
    [i, i + 1), so the centre of pixel i is at i + 0.5.
    """
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width] + 0.5
    expected = np.full(shape, BACKGROUND_COUNTS)
    for x, y, glyph in spots:
        expected += GLYPH_PEAK_COUNTS[glyph] * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * SPOT_SIGMA_PX**2)
        )
    return np.random.default_rng(seed).poisson(expected).astype(np.uint16)


def write_shared_memory_dump(directory: Path, frame: np.ndarray) -> None:
    """Write one frame in the layout the spotfinder's shared-memory reader takes.

    start_1 is the stream header, start_4 only has to exist, start_5 is the
    mask as int32 with 0 meaning a good pixel, and image_000000_2 is the frame
    as a bitshuffle/LZ4 chunk behind the 12-byte header HDF5's filter writes:
    the uncompressed size and the block size, both big-endian, with a block
    size of 0 meaning the default.
    """
    height, width = frame.shape
    header = {
        "nimages": 1,
        "ntrigger": 1,
        "y_pixels_in_detector": height,
        "x_pixels_in_detector": width,
        "bit_depth_image": 16,
        "countrate_correction_count_cutoff": TRUSTED_COUNT_CUTOFF,
        "wavelength": WAVELENGTH_A,
        "detector_distance": DETECTOR_DISTANCE_MM,
        "x_pixel_size": PIXEL_SIZE_M,
        "y_pixel_size": PIXEL_SIZE_M,
        "beam_center_x": width / 2,
        "beam_center_y": height / 2,
    }
    (directory / "start_1").write_text(json.dumps(header))
    (directory / "start_4").write_text("{}")
    np.zeros(frame.shape, dtype=np.int32).tofile(directory / "start_5")
    chunk = (
        struct.pack(">QI", frame.nbytes, 0)
        + bitshuffle.compress_lz4(frame, 0).tobytes()
    )
    (directory / "image_000000_2").write_bytes(chunk)


def find_spots(spotfinder: Path, frame: np.ndarray) -> list[tuple[float, float]]:
    """Run the spotfinder over one frame and return the spot centroids it reports."""
    with tempfile.TemporaryDirectory(prefix="ffs-banner-") as scratch:
        dump = Path(scratch)
        write_shared_memory_dump(dump, frame)

        # The spotfinder writes one JSON line per image to the pipe it is given
        read_fd, write_fd = os.pipe()
        command = [
            str(spotfinder),
            str(dump),
            "--images",
            "1",
            "--output-for-index",
            "--pipe_fd",
            str(write_fd),
        ]
        process = subprocess.Popen(
            command,
            pass_fds=[write_fd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        os.close(write_fd)
        with os.fdopen(read_fd) as pipe:
            lines = pipe.read().splitlines()
        log, _ = process.communicate()

    if process.returncode != 0 or not lines:
        raise RuntimeError(
            f"spotfinder exited {process.returncode} without reporting spots:\n"
            + log.decode(errors="replace")
        )
    centres = json.loads(lines[0])["spot_centers"]
    return [(centres[i], centres[i + 1]) for i in range(0, len(centres), 3)]


def measure_spots(frame: np.ndarray, centres: list[tuple[float, float]]) -> list[Spot]:
    """Attach to each centroid the background-subtracted counts in a box around it."""
    spots = []
    half = INTENSITY_BOX_HALF_PX
    for x, y in centres:
        column, row = int(x), int(y)
        box = frame[row - half : row + half + 1, column - half : column + half + 1]
        spots.append(Spot(x, y, float(box.sum() - BACKGROUND_COUNTS * box.size)))
    return spots


def colour_steps(ramp: tuple[str, ...]) -> list[str]:
    """Interpolate COLOUR_STEPS colours from the faintest spot colour to the brightest.

    The ramp's last colour is for background, so found spots run over the
    first three only.
    """
    stops = [
        tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5)) for colour in ramp[2::-1]
    ]
    steps = []
    for step in range(COLOUR_STEPS):
        position = step / (COLOUR_STEPS - 1) * (len(stops) - 1)
        low = min(int(position), len(stops) - 2)
        t = position - low
        rgb = [
            round(a + (b - a) * t)
            for a, b in zip(stops[low], stops[low + 1], strict=True)
        ]
        steps.append("#{:02X}{:02X}{:02X}".format(*rgb))
    return steps


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an RGBA image as a PNG, without depending on an imaging library."""
    height, width, _ = rgba.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    # Each scanline is prefixed with filter type 0, meaning none
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgba)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 9))
        + chunk(b"IEND", b"")
    )


def frame_png(frame: np.ndarray, colour: str) -> str:
    """Return the frame as a base64 PNG in one colour, with counts as opacity.

    Opacity follows log counts, so the background speckle and the faint
    background spots stay visible next to the strongest spots. It fades to
    nothing across the margin, so the frame has no hard edge against the page.
    """
    level = np.log1p(frame.astype(np.float64))
    height, width = frame.shape
    to_edge = np.minimum.outer(
        np.minimum(np.arange(height), np.arange(height)[::-1]),
        np.minimum(np.arange(width), np.arange(width)[::-1]),
    )
    fade = np.clip(to_edge / MARGIN_PX, 0.0, 1.0)
    fade = fade * fade * (3 - 2 * fade)
    alpha = np.round(level / level.max() * fade * FRAME_MAX_OPACITY * 255).astype(
        np.uint8
    )
    rgba = np.empty((*frame.shape, 4), dtype=np.uint8)
    rgba[..., :3] = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    rgba[..., 3] = alpha
    return base64.b64encode(png_bytes(rgba)).decode("ascii")


def build_svg(
    frame: np.ndarray,
    spots: list[Spot],
    palette: Palette,
    include_frame: bool,
    provenance: str,
) -> str:
    """Draw the found spots, and optionally the frame behind them, as an SVG."""
    height, width = frame.shape

    # Each spot gets a colour step and radius from its log intensity, scaled
    # between the faintest and the brightest found spot
    logs = np.log([max(spot.intensity, 1.0) for spot in spots])
    span = max(logs.max() - logs.min(), 1e-9)
    circles = []
    for spot, value in zip(spots, logs, strict=True):
        t = (value - logs.min()) / span
        step = round(t * (COLOUR_STEPS - 1))
        radius = SPOT_RADIUS_PX[0] + t * (SPOT_RADIUS_PX[1] - SPOT_RADIUS_PX[0])
        circles.append(
            f'<circle class="s{step}" cx="{spot.x:.2f}" cy="{spot.y:.2f}" r="{radius:.2f}"/>'
        )

    # One class per colour step, dark ramp by default and the light ramp when
    # the viewer prefers a light scheme
    dark, light = colour_steps(palette.dark), colour_steps(palette.light)
    dark_rules = " ".join(f".s{i}{{fill:{colour}}}" for i, colour in enumerate(dark))
    light_rules = " ".join(f".s{i}{{fill:{colour}}}" for i, colour in enumerate(light))
    style = [dark_rules, ".frame-light{display:none}"]
    style.append(
        "@media (prefers-color-scheme: light){"
        f"{light_rules} .frame-dark{{display:none}} .frame-light{{display:inline}}}}"
    )

    layers = []
    if include_frame:
        # The frame takes the ramp's background colour, the one the art uses
        # for its background glyph
        for scheme, ramp in (("dark", palette.dark), ("light", palette.light)):
            layers.append(
                f'<image class="frame-{scheme}" width="{width}" height="{height}" '
                f'image-rendering="pixelated" '
                f'href="data:image/png;base64,{frame_png(frame, ramp[2])}"/>'
            )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width * DISPLAY_SCALE:g}" height="{height * DISPLAY_SCALE:g}" '
            'role="img" aria-label="FAST, drawn in diffraction spots">',
            f"<metadata>{provenance}</metadata>",
            f"<style>{' '.join(style)}</style>",
            *layers,
            *circles,
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    parser.add_argument(
        "--palette",
        default="ice",
        help="colour map from banner.cc, by name (default: ice)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="SVG to write"
    )
    parser.add_argument(
        "--spotfinder",
        type=Path,
        default=DEFAULT_SPOTFINDER,
        help="spotfinder binary to run",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="seed for the noise"
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="draw the spots only, without the detector frame",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Art and colour maps both come from the C++ banner
    source = BANNER_SOURCE.read_text()
    art = read_banner_art(source)
    palettes = {palette.name.lower(): palette for palette in read_palettes(source)}
    palette = palettes.get(args.palette.lower())
    if palette is None:
        parser.error(
            f"unknown palette {args.palette!r}; choose from {', '.join(palettes)}"
        )

    # Render the art as a detector frame
    placed = art_spots(art)
    width = len(max(art, key=len)) * CELL_WIDTH_PX + 2 * MARGIN_PX
    height = len(art) * CELL_HEIGHT_PX + 2 * MARGIN_PX
    frame = render_frame((height, width), placed, args.seed)

    # Let the spotfinder decide what is a spot
    spots = measure_spots(frame, find_spots(args.spotfinder, frame))
    lettering = sum(1 for _, _, glyph in placed if glyph != BACKGROUND_GLYPH)
    logger.info(
        "Placed %d spots (%d lettering, %d background); the spotfinder found %d",
        len(placed),
        lettering,
        len(placed) - lettering,
        len(spots),
    )
    if len(spots) != lettering:
        logger.warning(
            "Found %d spots but the lettering has %d; check GLYPH_PEAK_COUNTS against "
            "the dispersion threshold",
            len(spots),
            lettering,
        )

    # Record where the picture came from inside it
    version = (
        subprocess.run(
            [str(args.spotfinder), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        .stdout.strip()
        .splitlines()
    )
    provenance = (
        f"Generated by scripts/banner_svg.py --palette {palette.name.lower()!r} "
        f"--seed {args.seed}{' --no-frame' if args.no_frame else ''}. "
        f"Spots found by spotfinder {version[-1] if version else '(unknown version)'}."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_svg(frame, spots, palette, not args.no_frame, provenance)
    )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
