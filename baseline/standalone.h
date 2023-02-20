#ifndef NO_TBX_H
#define NO_TBX_H

#ifdef USE_SPAN_BACKPORT
#include "span.hpp"
using tcb::span;
#else
#include <span>
using std::span;
#endif

#include <memory>
#include <type_traits>

template <typename T = double>
class StandaloneSpotfinder {
    // Make sure this is a type that we predeclare in the implementation
    static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "Only float or double is supported for Dials spotfinder internal implementation");
    // PIMPL-wrap the internals so that we don't need to include the algorithm here
    class StandaloneSpotfinderImpl;
    struct StandaloneSpotfinderImplDeleter {
        void operator()(StandaloneSpotfinderImpl*) const;
    };
    std::unique_ptr<StandaloneSpotfinderImpl, StandaloneSpotfinderImplDeleter> impl;

  public:
    StandaloneSpotfinder(size_t width, size_t height);

    auto standard_dispersion(const span<const T> image, const span<const bool> mask)
      -> span<const bool>;
    auto standard_dispersion(const span<const T> image, const span<const uint8_t> mask)
      -> span<const bool>;
};

#endif
