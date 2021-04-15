#ifndef BASELINE_H
#define BASELINE_H

#include "miniapp.h"

extern "C" {
void* create_spotfind(size_t width, size_t height);
void free_spotfind(void* context);
uint32_t spotfind_standard_dispersion(void* context, image_t image);
}
#endif