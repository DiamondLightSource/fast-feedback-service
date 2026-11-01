#!/usr/bin/env python3
"""
Draw the README banner from the spots the GPU spotfinder finds.

Renders the art in banner.cc as a detector frame, one Gaussian spot per
glyph on a Poisson background, and hands it to the spotfinder through
its shared-memory reader. The picture is what comes back: every pixel it
flagged as strong, coloured by its counts, and a red cross at each
centroid it reports. The background dots are too faint to make a spot,
so they only show in the frame behind.

The strong pixels come from the spotfinder's --writeout debug output,
so this breaks if that format changes. It needs a GPU and a built
spotfinder, so the SVG is committed and this is only rerun when the art
or a colour map changes.
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

# Half-length of each arm of the centroid cross, in pixels. Long enough to
# reach past the edge of the largest spots.
CENTROID_ARM_PX = 5.0

# Line width of the centroid crosses, in pixels
CENTROID_STROKE_PX = 0.5

# Colour of the centroid crosses in both schemes, the red of the spotfinder's
# own --writeout images
CENTROID_COLOUR = "#FF0000"

# Length of one loop of the cross animation, in seconds. The scan line adds
# each cross at the start of the loop and it fades away halfway through.
WAVE_CYCLE_S = 10.0

# Time for the scan line to cross the banner from top to bottom, in seconds
WAVE_SWEEP_S = 3.0

# Time for a cross to appear as the scan line reaches it, and to fade away
# half a loop later, in seconds
CROSS_FADE_IN_S = 0.3
CROSS_FADE_OUT_S = 1.0

# Height of the scan line, in pixels. It fades from clear at its trailing edge
# to SCAN_OPACITY at its leading edge, so the tail stays short.
SCAN_BAND_PX = 6.0
SCAN_OPACITY = 0.6

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


def find_spots(
    spotfinder: Path, frame: np.ndarray
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Run the spotfinder over one frame.

    Returns the spot centroids it reports and every pixel it flagged as
    strong, including those in groups too small to keep as a spot.
    """
    with tempfile.TemporaryDirectory(prefix="ffs-banner-") as scratch:
        dump = Path(scratch) / "dump"
        dump.mkdir()
        write_shared_memory_dump(dump, frame)

        # --writeout drops its debug files into the working directory
        work = Path(scratch) / "work"
        work.mkdir()

        # The spotfinder writes one JSON line per image to the pipe it is given
        read_fd, write_fd = os.pipe()
        command = [
            str(spotfinder),
            str(dump),
            "--images",
            "1",
            "--writeout",
            "--output-for-index",
            "--pipe_fd",
            str(write_fd),
        ]
        process = subprocess.Popen(
            command,
            cwd=work,
            pass_fds=[write_fd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        os.close(write_fd)
        with os.fdopen(read_fd) as pipe:
            lines = pipe.read().splitlines()
        log, _ = process.communicate()

        # One "x, y" line per strong pixel
        pixel_list = work / "pixels_00000.txt"
        strong = []
        if pixel_list.exists():
            for line in pixel_list.read_text().splitlines():
                x, y = line.split(",")
                strong.append((int(x), int(y)))

    if process.returncode != 0 or not lines or not strong:
        raise RuntimeError(
            f"spotfinder exited {process.returncode} without reporting spots:\n"
            + log.decode(errors="replace")
        )
    centres = json.loads(lines[0])["spot_centers"]
    return [(centres[i], centres[i + 1]) for i in range(0, len(centres), 3)], strong


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
    fade = edge_fade(frame.shape)
    alpha = np.round(level / level.max() * fade * FRAME_MAX_OPACITY * 255).astype(
        np.uint8
    )
    rgba = np.empty((*frame.shape, 4), dtype=np.uint8)
    rgba[..., :3] = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    rgba[..., 3] = alpha
    return base64.b64encode(png_bytes(rgba)).decode("ascii")


def edge_fade(shape: tuple[int, int]) -> np.ndarray:
    """Return 1 inside the margin falling to 0 at the image edge, as a smoothstep."""
    height, width = shape
    to_edge = np.minimum.outer(
        np.minimum(np.arange(height), np.arange(height)[::-1]),
        np.minimum(np.arange(width), np.arange(width)[::-1]),
    )
    fade = np.clip(to_edge / MARGIN_PX, 0.0, 1.0)
    return fade * fade * (3 - 2 * fade)


def edge_mask_png(shape: tuple[int, int]) -> str:
    """Return the edge fade as a base64 PNG for an SVG luminance mask."""
    grey = np.round(edge_fade(shape) * 255).astype(np.uint8)
    rgba = np.repeat(grey[..., None], 4, axis=2)
    rgba[..., 3] = 255
    return base64.b64encode(png_bytes(rgba)).decode("ascii")


def build_svg(
    frame: np.ndarray,
    strong: list[tuple[int, int]],
    centres: list[tuple[float, float]],
    palette: Palette,
    include_frame: bool,
    provenance: str,
) -> str:
    """Draw the strong pixels and centroids, and optionally the frame, as an SVG."""
    height, width = frame.shape

    # Each strong pixel gets a colour step from its log counts, scaled between
    # the faintest and the brightest strong pixel. One path per step keeps the
    # file small next to a rect per pixel.
    levels = np.log1p([float(frame[y, x]) for x, y in strong])
    span = max(levels.max() - levels.min(), 1e-9)
    outlines: dict[int, list[str]] = {}
    for (x, y), level in zip(strong, levels, strict=True):
        step = round((level - levels.min()) / span * (COLOUR_STEPS - 1))
        outlines.setdefault(step, []).append(f"M{x} {y}h1v1h-1z")
    pixel_paths = [
        f'<path class="s{step}" d="{"".join(outline)}"/>'
        for step, outline in sorted(outlines.items())
    ]

    # The scan line's leading edge runs from y = 0 to below the image over one
    # sweep, so anything at y is reached after that fraction of it
    sweep_distance = height + SCAN_BAND_PX

    def wave_delay(y: float) -> float:
        return y / sweep_distance * WAVE_SWEEP_S

    # The scan line, drawn with its leading edge on y = 0 and moved down by
    # the animation. Its gradient runs across the band, so it travels with
    # it. The mask sits on the group, not the moving polygon, so the line fades
    # out at the image edges the same way the frame does.
    scan = (
        f'<linearGradient id="scan-fade" gradientUnits="userSpaceOnUse" '
        f'x1="0" y1="{-SCAN_BAND_PX:g}" x2="0" y2="0">'
        f'<stop offset="0" stop-color="{CENTROID_COLOUR}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{CENTROID_COLOUR}" '
        f'stop-opacity="{SCAN_OPACITY:g}"/></linearGradient>'
        f'<mask id="edge-fade" maskUnits="userSpaceOnUse" x="0" y="0" '
        f'width="{width}" height="{height}"><image width="{width}" '
        f'height="{height}" href="data:image/png;base64,{edge_mask_png(frame.shape)}"/>'
        f'</mask><g mask="url(#edge-fade)">'
        f'<rect class="scan" fill="url(#scan-fade)" x="0" y="{-SCAN_BAND_PX:g}" '
        f'width="{width}" height="{SCAN_BAND_PX:g}"/>'
        "</g>"
    )

    # A cross at each centroid the spotfinder reported, added as the scan line
    # reaches it
    crosses = []
    for cx, cy in centres:
        delay = wave_delay(cy)
        crosses.append(
            f'<path style="animation-delay:{delay:.3f}s" '
            f'd="M{cx - CENTROID_ARM_PX:.2f} {cy:.2f}h{2 * CENTROID_ARM_PX:g}'
            f'M{cx:.2f} {cy - CENTROID_ARM_PX:.2f}v{2 * CENTROID_ARM_PX:g}"/>'
        )

    # Each cross fades in, holds until half the loop has passed since it
    # appeared, then fades away, so the crosses leave in the order they came.
    # Hidden before its delay has elapsed.
    fade_in = CROSS_FADE_IN_S / WAVE_CYCLE_S * 100
    fade_out = CROSS_FADE_OUT_S / WAVE_CYCLE_S * 100
    animation = (
        f"@keyframes wave{{0%{{opacity:0}} {fade_in:.2f}%{{opacity:1}} "
        f"50%{{opacity:1}} {50 + fade_out:.2f}%{{opacity:0}} 100%{{opacity:0}}}}"
        f".centroid path{{animation:wave {WAVE_CYCLE_S:g}s ease-in-out infinite "
        "backwards}"
    )

    # The scan line crosses once at the start of each loop and waits below the
    # image for the rest of it
    sweep = WAVE_SWEEP_S / WAVE_CYCLE_S * 100
    start, end = (
        "transform:translateY(0px)",
        f"transform:translateY({sweep_distance:g}px)",
    )
    animation += (
        f"@keyframes scan{{0%{{{start}}} {sweep:.2f}%{{{end}}} 100%{{{end}}}}}"
        f".scan{{animation:scan {WAVE_CYCLE_S:g}s linear infinite}}"
    )

    # Without motion, the crosses stay and the line is left out
    animation += (
        "@media (prefers-reduced-motion: reduce){"
        ".centroid path{animation:none;opacity:1}.scan{display:none}}"
    )

    # One class per colour step, dark ramp by default and the light ramp when
    # the viewer prefers a light scheme
    def rules(ramp: tuple[str, ...]) -> str:
        return " ".join(
            f".s{i}{{fill:{colour}}}" for i, colour in enumerate(colour_steps(ramp))
        )

    style = [
        rules(palette.dark),
        ".frame-light{display:none}",
        "@media (prefers-color-scheme: light){"
        f"{rules(palette.light)} .frame-dark{{display:none}} "
        ".frame-light{display:inline}}",
        animation,
    ]

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
            '<g shape-rendering="crispEdges">',
            *pixel_paths,
            "</g>",
            scan,
            f'<g class="centroid" fill="none" stroke="{CENTROID_COLOUR}" '
            f'stroke-width="{CENTROID_STROKE_PX:g}">',
            *crosses,
            "</g>",
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
    centres, strong = find_spots(args.spotfinder, frame)
    lettering = sum(1 for _, _, glyph in placed if glyph != BACKGROUND_GLYPH)
    logger.info(
        "Placed %d spots (%d lettering, %d background); the spotfinder found %d "
        "from %d strong pixels",
        len(placed),
        lettering,
        len(placed) - lettering,
        len(centres),
        len(strong),
    )
    if len(centres) != lettering:
        logger.warning(
            "Found %d spots but the lettering has %d; check GLYPH_PEAK_COUNTS against "
            "the dispersion threshold",
            len(centres),
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
        build_svg(frame, strong, centres, palette, not args.no_frame, provenance)
    )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
