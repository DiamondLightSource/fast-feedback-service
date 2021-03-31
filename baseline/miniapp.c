#include <stdio.h>
#include <stdlib.h>
#include "miniapp.h"

int main(int argc, char **argv) {
    if (argc == 2) {
        fprintf(stderr, "%s foobar.nxs foobar_000001.h5\n", argv[0]);
        return 1;
    }

    if (setup_hdf5_files(argv[1], argv[2]) < 0) {
        fprintf(stderr, "<shrug> bad thing </shrug>\n");
        exit(1);
    }

    size_t n_images = get_number_of_images();

    for (size_t j = 0; j < n_images; j++) {
      image_t image = get_image(j);

      size_t zero = 0;
      for (size_t i = 0; i < (image.fast * image.slow); i++) {
	if (image.data[i] == 0 && image.mask[i] == 1) {
	  zero ++;
	}
      }
      printf("image %ld had %ld valid zero pixels\n", j, zero);
	
      
      free_image(image);
    }


    
    return 0;
}
