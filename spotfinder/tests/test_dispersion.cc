/**
 * @file test_dispersion.cc
 * @brief Pins the dispersion thresholding decision to DIALS.
 *
 * Each case below is one neighbourhood, reduced to the moments the decision
 * takes, beside the verdict DIALS reaches on the same pixel. The values come
 * from DispersionThresholdDebug, which reports the background and signal
 * comparisons separately rather than only their conjunction, so each flag is
 * pinned on its own.
 *
 * Regenerate with:
 *
 *     module load dials/3.30
 *     dials.python spotfinder/tests/reference_dispersion.py
 *
 * The cases are centre pixels of 13x13 images whose (2*3+1)² window is fully
 * interior, so DIALS's edge clipping never enters and the moments here are the
 * whole window. DIALS's own defaults are used throughout: a (3, 3) kernel,
 * n_sig_b 6, n_sig_s 3, a global threshold of 0 and min_count 2. The GPU
 * instead runs min_count 3, which is tracked as a parity divergence, so these
 * tests state 2 explicitly rather than taking the service's value.
 */

#include <gtest/gtest.h>

#include <string>

#include "spotfinder/dispersion.hpp"

namespace {

constexpr uint8_t kDialsMinCount = 2;
constexpr float kDialsNSigB = 6.0f;
constexpr float kDialsNSigS = 3.0f;

struct NamedCase {
    const char *scenario;
    DispersionMoments moments;
    bool not_background;
    bool is_signal;
};

// DIALS 3.30.0.3-gdda5bf894-release
const NamedCase kNamedCases[] = {
  {"flat background of 2", {98, 196, 2, 49}, false, false},
  {"flat background with a bright centre", {146, 2692, 50, 49}, true, true},
  {"bright neighbour, quiet centre", {146, 2692, 2, 49}, true, false},
  {"all zero", {0, 0, 0, 49}, false, false},
  {"centre one count above a flat background", {99, 201, 3, 49}, false, false},
  {"a spot filling the window", {1960, 78400, 40, 49}, false, false},
  {"poisson-like background with a spot at the centre",
   {429, 38943, 180, 49},
   true,
   true},
  {"one neighbour just below the background bound", {113, 481, 2, 49}, false, false},
  {"one neighbour just above the background bound", {114, 516, 2, 49}, true, false},
  {"the three rows above the centre masked away", {114, 3708, 60, 28}, true, true},
  {"only two valid pixels, at min_count", {11, 85, 9, 2}, false, false},
  {"3 valid pixels, one neighbour at 12, just below the bound",
   {16, 152, 2, 3},
   false,
   false},
  {"3 valid pixels, one neighbour at 13, just above the bound",
   {17, 177, 2, 3},
   true,
   false},
  {"5 valid pixels, one neighbour at 12, just below the bound",
   {20, 160, 2, 5},
   false,
   false},
  {"5 valid pixels, one neighbour at 13, just above the bound",
   {21, 185, 2, 5},
   true,
   false},
  {"9 valid pixels, one neighbour at 15, just below the bound",
   {39, 297, 3, 9},
   false,
   false},
  {"9 valid pixels, one neighbour at 16, just above the bound",
   {40, 328, 3, 9},
   true,
   false},
  {"25 valid pixels, one neighbour at 15, just below the bound",
   {63, 321, 2, 25},
   false,
   false},
  {"25 valid pixels, one neighbour at 16, just above the bound",
   {64, 352, 2, 25},
   true,
   false},
  {"49 valid pixels, one neighbour at 17, just below the bound",
   {113, 481, 2, 49},
   false,
   false},
  {"49 valid pixels, one neighbour at 18, just above the bound",
   {114, 516, 2, 49},
   true,
   false},
};

// Neighbourhoods drawn at random and scored by the same DIALS call, so the
// table reaches cases nobody thought to write down. Eleven named cases do not
// pin the comparison tightly: an error that shifts the bound by the window's
// own sum slips past all of them, and lands inside this sample.
struct SampledCase {
    DispersionMoments moments;
    bool not_background;
    bool is_signal;
};

// DIALS 3.30.0.3-gdda5bf894-release
const SampledCase kSampledCases[] = {
  {{352, 111880, 18, 39}, true, false},  {{1296, 40220, 54, 43}, false, true},
  {{1456, 71480, 33, 44}, true, false},  {{234, 19032, 4, 46}, true, false},
  {{80, 194, 2, 43}, false, false},      {{313, 54935, 2, 41}, true, false},
  {{127, 421, 3, 45}, false, false},     {{1422, 61928, 60, 43}, true, true},
  {{2074, 227922, 39, 45}, true, false}, {{395, 156025, 0, 44}, true, false},
  {{229, 7541, 5, 44}, true, false},     {{1423, 45297, 59, 46}, false, true},
  {{75, 179, 0, 42}, false, false},      {{1271, 38345, 29, 43}, false, false},
  {{362, 3400, 20, 43}, false, true},    {{1410, 43936, 34, 46}, false, false},
  {{277, 17721, 0, 46}, true, false},    {{614, 134018, 2, 39}, true, false},
  {{617, 164013, 2, 44}, true, false},   {{257, 49045, 39, 45}, true, true},
  {{385, 7035, 66, 41}, true, true},     {{81, 175, 2, 42}, false, false},
  {{118, 1118, 30, 42}, true, true},     {{0, 0, 0, 43}, false, false},
  {{283, 43851, 0, 42}, true, false},    {{1667, 112309, 29, 45}, true, false},
  {{121, 429, 3, 43}, false, false},     {{418, 9564, 6, 44}, true, false},
  {{1838, 252070, 84, 41}, true, true},  {{709, 144823, 7, 44}, true, false},
  {{1451, 54329, 33, 44}, true, false},  {{185, 3797, 58, 43}, true, true},
  {{370, 37290, 61, 45}, true, true},    {{1548, 140156, 60, 38}, true, true},
  {{113, 1229, 32, 44}, true, true},     {{0, 0, 0, 46}, false, false},
  {{190, 6058, 56, 41}, true, true},     {{460, 140076, 2, 46}, true, false},
  {{0, 0, 0, 46}, false, false},         {{1316, 52772, 49, 40}, true, false},
  {{1695, 158183, 23, 44}, true, false}, {{389, 6395, 60, 42}, true, true},
  {{352, 51106, 50, 44}, true, true},    {{1583, 196705, 37, 40}, true, false},
  {{367, 3165, 6, 46}, false, false},    {{135, 455, 4, 45}, false, false},
  {{373, 3257, 13, 45}, false, false},   {{423, 86333, 4, 44}, true, false},
  {{246, 27936, 51, 42}, true, true},    {{810, 206662, 39, 44}, true, true},
  {{367, 51223, 13, 46}, true, false},   {{1468, 47836, 45, 46}, false, false},
  {{36, 48, 2, 42}, false, false},       {{253, 60089, 8, 44}, true, false},
  {{121, 2393, 47, 40}, true, true},     {{1242, 37378, 40, 42}, false, false},
  {{52, 86, 2, 47}, false, false},       {{745, 245671, 1, 44}, true, false},
  {{904, 239250, 3, 43}, true, false},   {{1727, 197005, 33, 45}, true, false},
  {{143, 589, 10, 41}, false, true},     {{72, 912, 29, 39}, true, true},
  {{1386, 43288, 34, 45}, false, false}, {{449, 68739, 2, 45}, true, false},
  {{1300, 38932, 28, 44}, false, false}, {{0, 0, 0, 40}, false, false},
  {{138, 2452, 47, 42}, true, true},     {{1719, 205405, 53, 44}, true, false},
  {{0, 0, 0, 38}, false, false},         {{78, 1230, 34, 39}, true, true},
  {{1699, 198319, 29, 44}, true, false}, {{402, 57782, 61, 43}, true, true},
  {{381, 5361, 50, 42}, true, true},     {{1341, 43803, 70, 43}, false, true},
  {{684, 102848, 144, 44}, true, true},  {{347, 2925, 11, 43}, false, false},
  {{190, 19958, 1, 44}, true, false},    {{362, 30800, 61, 47}, true, true},
  {{341, 2939, 10, 42}, false, false},   {{381, 3439, 7, 44}, false, false},
  {{0, 0, 0, 44}, false, false},         {{272, 51600, 0, 42}, true, false},
  {{347, 2887, 9, 45}, false, false},    {{342, 2896, 8, 42}, false, false},
  {{20, 400, 20, 40}, true, true},       {{74, 190, 3, 39}, false, false},
  {{1535, 103533, 48, 44}, true, false}, {{135, 1039, 26, 38}, true, true},
  {{206, 3684, 56, 47}, true, true},     {{150, 3256, 55, 45}, true, true},
  {{514, 38796, 9, 42}, true, false},    {{701, 146961, 5, 43}, true, false},
  {{274, 20072, 4, 45}, true, false},    {{1326, 40676, 31, 44}, false, false},
  {{228, 13170, 4, 40}, true, false},    {{1378, 46552, 82, 44}, true, true},
  {{132, 470, 3, 44}, false, false},     {{132, 470, 3, 45}, false, false},
  {{674, 109908, 4, 45}, true, false},   {{1241, 38123, 32, 41}, false, false},
  {{565, 57493, 5, 44}, true, false},    {{137, 465, 4, 45}, false, false},
  {{1712, 161380, 218, 42}, true, true}, {{498, 38828, 7, 38}, true, false},
  {{141, 657, 15, 43}, false, true},     {{476, 21076, 11, 44}, true, false},
  {{324, 59720, 2, 44}, true, false},    {{581, 135781, 25, 45}, true, true},
  {{1226, 36234, 26, 42}, false, false}, {{507, 109877, 42, 41}, true, true},
  {{171, 1867, 37, 44}, true, true},     {{386, 114322, 2, 40}, true, false},
  {{1201, 275633, 10, 44}, true, false}, {{133, 467, 4, 44}, false, false},
  {{191, 21853, 53, 45}, true, true},    {{185, 9249, 3, 44}, true, false},
  {{1175, 35739, 27, 39}, false, false}, {{82, 204, 2, 41}, false, false},
  {{754, 266772, 28, 42}, true, false},  {{1225, 38119, 30, 40}, false, false},
  {{270, 19238, 3, 45}, true, false},    {{1062, 183356, 11, 44}, true, false},
  {{0, 0, 0, 36}, false, false},         {{92, 232, 1, 44}, false, false},
  {{1615, 116865, 75, 44}, true, true},  {{91, 2189, 4, 44}, true, false},
  {{173, 3945, 39, 45}, true, true},     {{200, 24418, 53, 40}, true, true},
  {{280, 57182, 0, 42}, true, false},    {{774, 152732, 36, 43}, true, true},
  {{265, 20017, 3, 46}, true, false},    {{1463, 46275, 30, 47}, false, false},
  {{85, 263, 8, 39}, false, true},       {{361, 3599, 25, 42}, false, true},
  {{170, 3620, 57, 42}, true, true},     {{682, 132254, 7, 42}, true, false},
  {{121, 973, 27, 44}, true, true},      {{372, 6312, 60, 39}, true, true},
  {{512, 19900, 62, 44}, true, true},    {{631, 59763, 8, 40}, true, false},
  {{432, 150434, 47, 44}, true, true},   {{385, 88417, 3, 45}, true, false},
  {{416, 97078, 28, 42}, true, true},    {{0, 0, 0, 45}, false, false},
  {{0, 0, 0, 45}, false, false},         {{343, 2769, 10, 46}, false, false},
  {{230, 5280, 23, 45}, true, true},     {{140, 458, 3, 47}, false, false},
  {{1272, 38484, 20, 43}, false, false}, {{1405, 43799, 49, 46}, false, true},
};

constexpr DispersionImpl kBothForms[] = {DispersionImpl::FastInteger,
                                         DispersionImpl::DialsDivisionFree};

DispersionFlags evaluate(const NamedCase &named, DispersionImpl impl) {
    return dispersion_flags(
      named.moments, kDialsMinCount, kDialsNSigB, kDialsNSigS, 0.0f, impl);
}

}  // namespace

// Every case, including the pairs one count either side of the background
// bound at five window sizes, which is where a rearranged comparison goes
// wrong first. Both forms have to clear it: the point of carrying two is that
// the choice is a cost decision, not a numerical one.
TEST(DispersionThreshold, BothFormsMatchDialsOnEveryCase) {
    for (const auto impl : kBothForms) {
        for (const auto &named : kNamedCases) {
            SCOPED_TRACE(named.scenario);
            const auto flags = evaluate(named, impl);
            EXPECT_EQ(flags.not_background, named.not_background)
              << "background flag disagrees with DIALS";
            EXPECT_EQ(flags.is_signal, named.is_signal)
              << "signal flag disagrees with DIALS";
        }
    }
}

// The sample is what actually constrains the arithmetic: the comparison has to
// land on DIALS's answer for all of it, not just for the cases chosen to be
// legible.
TEST(DispersionThreshold, BothFormsMatchDialsOverTheSample) {
    for (const auto impl : kBothForms) {
        int index = 0;
        for (const auto &sampled : kSampledCases) {
            SCOPED_TRACE("sampled case " + std::to_string(index++));
            const auto flags = dispersion_flags(
              sampled.moments, kDialsMinCount, kDialsNSigB, kDialsNSigS, 0.0f, impl);
            EXPECT_EQ(flags.not_background, sampled.not_background)
              << "background flag disagrees with DIALS";
            EXPECT_EQ(flags.is_signal, sampled.is_signal)
              << "signal flag disagrees with DIALS";
        }
    }
}

// A neighbourhood thinner than min_count carries no usable sampling
// deviation, and the service asks for 3 where DIALS asks for 2, so the
// two-pixel case has to come back empty at the service's setting.
TEST(DispersionThreshold, ReportsNeitherFlagBelowMinCount) {
    const DispersionMoments two_pixels{11, 85, 9, 2};
    for (const auto impl : kBothForms) {
        const auto flags =
          dispersion_flags(two_pixels, 3, kDialsNSigB, kDialsNSigS, 0.0f, impl);
        EXPECT_FALSE(flags.not_background) << "background flag set below min_count";
        EXPECT_FALSE(flags.is_signal) << "signal flag set below min_count";
    }
}

// The global threshold admits a count strictly above it, so the centre pixel's
// own value is the first one rejected. Raising it must suppress the signal flag
// and leave the background flag alone.
TEST(DispersionThreshold, GlobalThresholdSuppressesTheSignalFlag) {
    const DispersionMoments spot{429, 38943, 180, 49};
    for (const auto impl : kBothForms) {
        const auto below = dispersion_flags(
          spot, kDialsMinCount, kDialsNSigB, kDialsNSigS, 179.0f, impl);
        EXPECT_TRUE(below.is_signal) << "a count above the threshold was rejected";
        const auto equal = dispersion_flags(
          spot, kDialsMinCount, kDialsNSigB, kDialsNSigS, 180.0f, impl);
        EXPECT_FALSE(equal.is_signal) << "a count equal to the threshold was accepted";
        EXPECT_TRUE(equal.not_background)
          << "the global threshold reached the background flag";
    }
}
