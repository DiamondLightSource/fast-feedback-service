/**
 * @file banner.hpp
 * @brief Startup banner: "FAST" drawn as a diffraction pattern, the letters
 *        made of Bragg spots on a lattice.
 *
 * Spots are coloured by intensity, with the colour map picked at random each run.
 *
 * The art was generated once by rendering "FAST" with pyfiglet and replacing
 * each filled cell of the lettering with a spot of random brightness, then
 * tidied by hand.
 */

#pragma once

#include <fmt/color.h>
#include <fmt/core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>

// The frozen spot art, 63 columns by 7 rows. The leading newline only keeps the
// raw string aligned in the source and is not printed.
inline constexpr std::string_view banner_art = R"(
            ·
• ● ∙ • ● ● ●     • ● ● ● ●     ● • ∙ • ∙ • ●   ● ● • • ● • • ●
• ∙             • ●       ● •   ● ∙                   • ●
● • • • •       • ● ● ∙ ● ● ●   • ● • • ● • ●         ● ●
• ●             • ●       ● •             ● •         • •
● •             ● ●       ● ●   ● ∙ ● ● ∙ • ∙         • •
                        ·
)";

namespace banner_detail {

// Spot glyphs
inline constexpr std::array<std::string_view, 4> spot_glyphs{"●", "•", "∙", "·"};

/**
 * @brief One intensity ramp, brightest spot to background spot.
 */
struct Ramp {
    std::array<fmt::rgb, 4> truecolor;         ///< 24-bit colours
    std::array<fmt::terminal_color, 4> basic;  ///< Nearest 16-colour stand-ins
};

/**
 * @brief A colour map as two ramps, one per terminal background.
 *
 * On a dark background the strongest spot is the lightest colour; on a light
 * background it is the darkest.
 */
struct Palette {
    Ramp dark;   ///< Ramp for a dark terminal background
    Ramp light;  ///< Ramp for a light terminal background
};

// Candidate colour maps, one picked at random per run.
inline constexpr std::array<Palette, 5> palettes{{
  // Ice: white-hot core fading through cyan to deep blue; navy core on light
  {{{fmt::rgb(0xFFFFFF), fmt::rgb(0x5FD7FF), fmt::rgb(0x1F6FBF), fmt::rgb(0x3A4A5A)},
    {fmt::terminal_color::bright_white,
     fmt::terminal_color::bright_cyan,
     fmt::terminal_color::blue,
     fmt::terminal_color::bright_black}},
   {{fmt::rgb(0x08306B), fmt::rgb(0x1F78B4), fmt::rgb(0x6BAED6), fmt::rgb(0xC6DBEF)},
    {fmt::terminal_color::blue,
     fmt::terminal_color::cyan,
     fmt::terminal_color::bright_cyan,
     fmt::terminal_color::bright_black}}},
  // Heat: pale yellow through orange to deep red, as in the DIALS image viewer
  {{{fmt::rgb(0xFFF6C8), fmt::rgb(0xFF9A1F), fmt::rgb(0xC8261E), fmt::rgb(0x5A2A1A)},
    {fmt::terminal_color::bright_yellow,
     fmt::terminal_color::yellow,
     fmt::terminal_color::red,
     fmt::terminal_color::bright_black}},
   {{fmt::rgb(0x7F0000), fmt::rgb(0xD7301F), fmt::rgb(0xFC8D59), fmt::rgb(0xF5D5C0)},
    {fmt::terminal_color::red,
     fmt::terminal_color::bright_red,
     fmt::terminal_color::yellow,
     fmt::terminal_color::bright_black}}},
  // Greyscale: a raw detector frame
  {{{fmt::rgb(0xFFFFFF), fmt::rgb(0xB0B0B0), fmt::rgb(0x6A6A6A), fmt::rgb(0x3A3A3A)},
    {fmt::terminal_color::bright_white,
     fmt::terminal_color::white,
     fmt::terminal_color::bright_black,
     fmt::terminal_color::bright_black}},
   {{fmt::rgb(0x000000), fmt::rgb(0x4A4A4A), fmt::rgb(0x8A8A8A), fmt::rgb(0xC8C8C8)},
    {fmt::terminal_color::black,
     fmt::terminal_color::bright_black,
     fmt::terminal_color::bright_black,
     fmt::terminal_color::white}}},
  // Phosphor: green CRT
  {{{fmt::rgb(0xC8FFC8), fmt::rgb(0x39FF14), fmt::rgb(0x1A8F1A), fmt::rgb(0x1F3F1F)},
    {fmt::terminal_color::bright_white,
     fmt::terminal_color::bright_green,
     fmt::terminal_color::green,
     fmt::terminal_color::bright_black}},
   {{fmt::rgb(0x00441B), fmt::rgb(0x238B45), fmt::rgb(0x74C476), fmt::rgb(0xC7E9C0)},
    {fmt::terminal_color::green,
     fmt::terminal_color::bright_green,
     fmt::terminal_color::bright_green,
     fmt::terminal_color::bright_black}}},
  // X-ray blue: a single hue at falling intensity
  {{{fmt::rgb(0xE8F4FF), fmt::rgb(0x7FB8FF), fmt::rgb(0x3A6FD8), fmt::rgb(0x2A3A5A)},
    {fmt::terminal_color::bright_white,
     fmt::terminal_color::bright_blue,
     fmt::terminal_color::blue,
     fmt::terminal_color::bright_black}},
   {{fmt::rgb(0x0B1F5C), fmt::rgb(0x2B5CC8), fmt::rgb(0x7FA8E8), fmt::rgb(0xC8DAF5)},
    {fmt::terminal_color::blue,
     fmt::terminal_color::bright_blue,
     fmt::terminal_color::bright_blue,
     fmt::terminal_color::bright_black}}},
}};

/**
 * @brief How much colour the terminal can take.
 */
enum class ColourMode : uint8_t { None, Basic, Truecolor };

/**
 * @brief Read the colour support from NO_COLOR and COLORTERM.
 *
 * @return None if NO_COLOR is set, Truecolor if COLORTERM advertises 24-bit
 *         support, Basic otherwise
 */
inline ColourMode detect_colour_mode() {
    // NO_COLOR disables colour whenever it is set and non-empty
    const char *no_colour = std::getenv("NO_COLOR");
    if (no_colour != nullptr && no_colour[0] != '\0') {
        return ColourMode::None;
    }
    // COLORTERM is the de facto advertisement of 24-bit support
    const char *colorterm = std::getenv("COLORTERM");
    if (colorterm != nullptr) {
        std::string_view value{colorterm};
        if (value == "truecolor" || value == "24bit") {
            return ColourMode::Truecolor;
        }
    }
    return ColourMode::Basic;
}

/**
 * @brief Whether the terminal background is dark or light.
 */
enum class Background : uint8_t { Dark, Light };

/**
 * @brief Read the terminal background from COLORFGBG.
 *
 * COLORFGBG is "fg;bg" or "fg;default;bg", and only rxvt, Konsole and a few
 * others set it. Following vim, background colours 0-6 and 8 are dark and the
 * rest light.
 *
 * @return Light if COLORFGBG names a light background, Dark otherwise,
 *         including when it is unset
 */
inline Background detect_background() {
    const char *colorfgbg = std::getenv("COLORFGBG");
    if (colorfgbg == nullptr) {
        return Background::Dark;
    }
    // The background is the last field
    std::string_view value{colorfgbg};
    const auto sep = value.rfind(';');
    if (sep == std::string_view::npos) {
        return Background::Dark;
    }
    const std::string_view bg = value.substr(sep + 1);
    const bool dark = (bg.size() == 1 && bg[0] >= '0' && bg[0] <= '6') || bg == "8";
    return dark || bg.empty() || bg == "default" ? Background::Dark : Background::Light;
}

}  // namespace banner_detail

/**
 * @brief Print the startup banner and a line naming the program to stdout.
 *
 * @param program Name shown under the art, e.g. "spotfinder".
 */
inline void print_banner(std::string_view program) {
    using namespace banner_detail;

    // Resolve one text style per glyph for this run. An empty style emits no
    // escape codes, which covers NO_COLOR without a separate code path.
    std::array<fmt::text_style, spot_glyphs.size()> styles{};
    const ColourMode mode = detect_colour_mode();
    if (mode != ColourMode::None) {
        std::random_device entropy;
        std::uniform_int_distribution<size_t> pick(0, palettes.size() - 1);
        const Palette &palette = palettes[pick(entropy)];
        const Ramp &ramp =
          detect_background() == Background::Light ? palette.light : palette.dark;
        for (size_t i = 0; i < styles.size(); ++i) {
            styles[i] = mode == ColourMode::Truecolor ? fmt::fg(ramp.truecolor[i])
                                                      : fmt::fg(ramp.basic[i]);
        }
    }

    // Walk the art, colouring each spot glyph and passing everything else
    // through. Glyphs are multi-byte UTF-8, so match by prefix, not by char.
    std::string_view art = banner_art.substr(1);
    while (!art.empty()) {
        size_t match = 0;
        while (match < spot_glyphs.size() && !art.starts_with(spot_glyphs[match])) {
            ++match;
        }
        if (match == spot_glyphs.size()) {
            std::fputc(art.front(), stdout);
            art.remove_prefix(1);
            continue;
        }
        fmt::print(styles[match], "{}", spot_glyphs[match]);
        art.remove_prefix(spot_glyphs[match].size());
    }

    // Program name under the art in the palette's medium-intensity colour
    const fmt::text_style bold =
      mode == ColourMode::None ? fmt::text_style{} : fmt::emphasis::bold;
    fmt::print(
      "{} · {}\n\n", fmt::styled("FAST", bold), fmt::styled(program, styles[1]));
    std::fflush(stdout);
}
