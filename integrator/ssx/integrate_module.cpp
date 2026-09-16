#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ssx_integrate.hpp"


NB_MODULE(integrate, m) {
    m.def("ssx_integrate", &ssx_integrate, "ssx integrate");
}