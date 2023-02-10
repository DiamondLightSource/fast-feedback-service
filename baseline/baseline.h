#ifndef BASELINE_H
#define BASELINE_H

#include "h5read.h"

#ifdef __cplusplus
extern "C" {
#endif
void* spotfinder_create(size_t width, size_t height);
void spotfinder_free(void* context);
uint32_t spotfinder_standard_dispersion(void* context,
                                        image_t* image,
                                        bool** destination = nullptr);
#ifdef __cplusplus
}
#endif

#endif