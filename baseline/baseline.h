#ifndef BASELINE_H
#define BASELINE_H

#include "miniapp.h"

extern "C" {
void* create_spotfinder(size_t width, size_t height);
void free_spotfinder(void* context);
uint32_t spotfinder_standard_dispersion(void* context, image_t* image);
}
#endif