"""Emit reference dispersion-threshold values from DIALS for test_dispersion.cc.

Each case is a 13x13 image whose centre pixel carries a fully interior 7x7
window, so DIALS's edge clipping never enters. The window's moments are
computed here from the same arrays DIALS is given, and DIALS's own per-flag
verdicts come from DispersionThresholdDebug.
"""

from __future__ import annotations

import random

from dials.algorithms.image.threshold import DispersionThresholdDebug
from dials.array_family import flex
from dials.util.version import dials_version

SIZE = 13
CENTRE = SIZE // 2
KERNEL = 3  # DIALS kernel_size, giving a (2*3+1)² window
NSIG_B = 6.0
NSIG_S = 3.0
THRESHOLD = 0.0
MIN_COUNT = 2


def case(name, pixels, masked=()):
    """pixels: dict of (y, x) -> value over a background of 2, unless given."""
    image = flex.double(flex.grid(SIZE, SIZE), float(pixels.pop("background", 2)))
    for (y, x), v in pixels.items():
        image[y, x] = float(v)
    mask = flex.bool(flex.grid(SIZE, SIZE), True)
    for y, x in masked:
        mask[y, x] = False

    debug = DispersionThresholdDebug(
        image, mask, (KERNEL, KERNEL), NSIG_B, NSIG_S, THRESHOLD, MIN_COUNT
    )

    # The moments of the window DIALS summed, counting only unmasked pixels.
    n = total = total_sq = 0
    for dy in range(-KERNEL, KERNEL + 1):
        for dx in range(-KERNEL, KERNEL + 1):
            y, x = CENTRE + dy, CENTRE + dx
            if not mask[y, x]:
                continue
            v = int(image[y, x])
            n += 1
            total += v
            total_sq += v * v

    return {
        "name": name,
        "n": n,
        "sum": total,
        "sumsq": total_sq,
        "this_pixel": int(image[CENTRE, CENTRE]),
        "cv": bool(debug.cv_mask()[CENTRE, CENTRE]),
        "value": bool(debug.value_mask()[CENTRE, CENTRE]),
        "final": bool(debug.final_mask()[CENTRE, CENTRE]),
        "mean": debug.mean()[CENTRE, CENTRE],
        "variance": debug.variance()[CENTRE, CENTRE],
    }


cases = [
    case("flat background of 2", {}),
    case("flat background with a bright centre", {(CENTRE, CENTRE): 50}),
    case("bright neighbour, quiet centre", {(CENTRE, CENTRE + 2): 50}),
    case("all zero", {"background": 0}),
    case("centre one count above a flat background", {(CENTRE, CENTRE): 3}),
    case("a spot filling the window", {"background": 40}),
    case(
        "poisson-like background with a spot at the centre",
        {
            (CENTRE, CENTRE): 180,
            (CENTRE - 1, CENTRE): 40,
            (CENTRE + 1, CENTRE): 36,
            (CENTRE, CENTRE - 1): 44,
            (CENTRE, CENTRE + 1): 39,
            (CENTRE - 2, CENTRE + 1): 1,
            (CENTRE + 2, CENTRE - 2): 4,
            (CENTRE - 3, CENTRE - 3): 3,
        },
    ),
    # 17 and 18 straddle the background bound for a 49-pixel window, so this
    # pair is what catches a rearrangement that shifts the comparison.
    case("one neighbour just below the background bound", {(CENTRE + 2, CENTRE): 17}),
    case("one neighbour just above the background bound", {(CENTRE + 2, CENTRE): 18}),
    case(
        "the three rows above the centre masked away",
        {(CENTRE, CENTRE): 60},
        masked=[(y, x) for y in range(CENTRE - 3, CENTRE) for x in range(SIZE)],
    ),
    case(
        "only two valid pixels, at min_count",
        {(CENTRE, CENTRE): 9},
        masked=[
            (y, x)
            for y in range(CENTRE - 3, CENTRE + 4)
            for x in range(CENTRE - 3, CENTRE + 4)
            if not ((y, x) == (CENTRE, CENTRE) or (y, x) == (CENTRE, CENTRE + 1))
        ],
    ),
]


def random_cases(count, seed=67):
    """Neighbourhoods drawn at random, so the table is not only the cases
    someone thought to write down. The mix is a quiet Poisson background with
    occasional bright pixels and occasional masked ones, which puts a useful
    share of the sample near one bound or the other."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        rate = rng.choice([0, 1, 2, 3, 8, 30])
        pixels = {"background": 0}
        for y in range(SIZE):
            for x in range(SIZE):
                v = sum(rng.random() < 0.5 for _ in range(2 * rate)) if rate else 0
                if rng.random() < 0.02:
                    v += rng.randint(5, 400)
                pixels[(y, x)] = v
        # The centre pixel is brightened deliberately in some cases, since a
        # background drawn at random almost never puts the bright pixel where
        # the signal test looks.
        if rng.random() < 0.45:
            pixels[(CENTRE, CENTRE)] = rng.randint(rate + 1, rate + 60)
        masked = [
            (y, x)
            for y in range(SIZE)
            for x in range(SIZE)
            if rng.random() < rng.choice([0.0, 0.0, 0.1, 0.4])
            and (y, x) != (CENTRE, CENTRE)
        ]
        out.append(case(f"random {i}", pixels, masked))
    return out


def crossing_pairs():
    """The two adjacent configurations either side of the background bound, for
    several window sizes.

    The bound carries a √(2/(n-1)) factor, so its shape in n is only pinned by
    cases that sit against it at more than one n. These are found by searching
    for the outlier value where DIALS changes its mind, and both sides are
    emitted."""
    out = []
    for valid, background in ((3, 2), (5, 2), (9, 3), (25, 2), (49, 2)):
        # Mask the window down to `valid` pixels, keeping the centre.
        window = [
            (y, x)
            for y in range(CENTRE - KERNEL, CENTRE + KERNEL + 1)
            for x in range(CENTRE - KERNEL, CENTRE + KERNEL + 1)
        ]
        window.remove((CENTRE, CENTRE))
        keep = set(window[: valid - 1]) | {(CENTRE, CENTRE)}
        masked = [c for c in window if c not in keep]
        outlier = sorted(keep - {(CENTRE, CENTRE)})[0]

        def verdict(value):
            return case("", {"background": background, outlier: value}, masked)

        low, high = background, background + 4000
        if verdict(high)["cv"] == verdict(low)["cv"]:
            continue
        while high - low > 1:
            mid = (low + high) // 2
            if verdict(mid)["cv"] == verdict(low)["cv"]:
                low = mid
            else:
                high = mid
        for value, side in ((low, "below"), (high, "above")):
            c = verdict(value)
            c["name"] = (
                f"{valid} valid pixels, one neighbour at {value}, just {side} the bound"
            )
            out.append(c)
    return out


print(f"// {dials_version()}")
print(
    f"// kernel_size ({KERNEL}, {KERNEL}), n_sig_b {NSIG_B}, n_sig_s {NSIG_S}, "
    f"threshold {THRESHOLD}, min_count {MIN_COUNT}"
)
for c in cases + crossing_pairs():
    print(
        '    {{"{name}", {{{sum}, {sumsq}, {this_pixel}, {n}}}, {cv}, {value}}},'
        "  // mean {mean:.4f} var {variance:.4f} final {final}".format(
            **{**c, "cv": str(c["cv"]).lower(), "value": str(c["value"]).lower()}
        )
    )

print()
print(
    f"// {dials_version()}, {len(cases)} named cases above and the rest drawn at random"
)
for c in random_cases(150):
    print(
        "    {{{{{sum}, {sumsq}, {this_pixel}, {n}}}, {cv}, {value}}},".format(
            **{**c, "cv": str(c["cv"]).lower(), "value": str(c["value"]).lower()}
        )
    )
