#ifndef __MINIAPP_H
#define __MINIAPP_H
#include <unistd.h>

typedef struct image {
    uint16_t *data;
    uint8_t *mask;
    size_t n_slow;
    size_t n_fast;
} image;

#endif
