#ifndef SPOTFIND_H
#define SPOTFIND_H

#include <algorithm>
#include <iostream>
#include <random>

// #include "TinyTIFF/tinytiffwriter.h"

#include <scitbx/array_family/accessors/c_grid.h>
#include <scitbx/array_family/shared.h>

#include "local.h"
// #include "baseline.h"

namespace af = scitbx::af;

const size_t IMAGE_W = 4000;
const size_t IMAGE_H = 4000;

const af::tiny<int, 2> kernel_size_(3, 3);
const af::tiny<int, 2> image_size_(IMAGE_W, IMAGE_H);

const int min_count_ = 2;
const double threshold_ = 0.0;
const double nsig_b_ = 6.0;
const double nsig_s_ = 3.0;

#endif
