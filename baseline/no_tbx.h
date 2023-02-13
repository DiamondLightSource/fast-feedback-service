#ifndef NO_TBX_H
#define NO_TBX_H

#include "h5read.h"

void* no_tbx_spotfinder_create(size_t width, size_t height);
void no_tbx_spotfinder_free(void* context);
uint32_t no_tbx_spotfinder_standard_dispersion(void* context,
                                               image_t* image,
                                               bool** destination = nullptr);

#endif