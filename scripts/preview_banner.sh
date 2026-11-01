#!/usr/bin/env bash
# Print the startup banner in every palette, on a dark and a light background.
#
# FFS_BANNER_PALETTE pins the palette by its index, and COLORFGBG sets the
# background the banner assumes ("15;0" dark, "0;15" light). script runs the
# program under a pseudo-terminal, so the banner's TTY check passes inside the
# loop. Its stdin is /dev/null, since script switches a terminal stdin to raw
# mode and leaves it that way if killed, and sed reads to the end rather than
# closing the pipe early as head would.
#
#   scripts/preview_banner.sh                  uses build/bin/spotfinder
#   scripts/preview_banner.sh path/to/binary   any program that prints the banner
#
# The index list follows the palette table in src/ffs/banner.cc.
set -euo pipefail

binary="${1:-build/bin/spotfinder}"

for palette in 0 1 2 3 4; do
    for background in "15;0" "0;15"; do
        echo "palette $palette, COLORFGBG=$background"
        FFS_BANNER_PALETTE=$palette COLORFGBG=$background \
            script -qc "$binary --help" /dev/null </dev/null | sed -n '1,9p'
    done
done
