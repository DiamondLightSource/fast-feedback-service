# Running the service

The service, a Python script in [`src/ffs/`], watches a queue for requests
to process images. It then launches the compiled spotfinder to process the
images, built from the source in [`spotfinder/`].

## Environment Variables
The service uses the following environment variables:
- `SPOTFINDER`: The path to the compiled spotfinder executable.
  - If not set, it is looked up on `PATH`, where an install puts it. Point
    this at `build/bin/spotfinder` to use a development build instead.
- `LOG_LEVEL`: The level of logging to use provided by `spdlog`. Not setting this will default to `info`.
  - Other levels are: `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off`.

## Launching under Zocalo
To run the service, you need to be on a machine with an NVIDIA GPU and the CUDA toolkit installed.

Set up the environment variables:
```bash
export SPOTFINDER=/path/to/spotfinder
export ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml
```
Then launch the service through Zocalo:
```bash
zocalo.service -s GPUPerImageAnalysis
```

## CBF requests
As well as NeXus/HDF5 (and `/dev/shm` streams), a request `filename` may be
CBF: either a single `.cbf` image, or a template with a run of `#` standing
for the image number (e.g. `image_#####.cbf`).

- Images are spotfound with the 32-bit spotfinder, and indexed in the same
  way as HDF5 data when `unit_cell` and a wavelength are available.
- Detector type, pixel size, image size and sensor are read from the CBF
  header (`detector` in the request overrides only the type). A missing
  `wavelength` falls back to the header value. Beam centre and distance still
  come from the request.
- The result `file-number` is `start_frame_index + 1` for the first image of
  the request, so `start_frame_index` is the zero-based position in the
  collection. For templates, `file` in the result is the actual image file.
- A single file is always one image; `number_of_frames` is ignored.
  Message ordering is skipped for single files, and the "too old" check is
  skipped for all CBF.
- For a template, the file number of the first image of the collection is
  `first_image_number`. If this isn't given, it is guessed as the lowest
  numbered file on disk matching the template (with the same number of
  digits), waiting up to 60 s for one to appear. Image `start_frame_index`
  of the request is then file `first_image_number + start_frame_index`.
  The guess is wrong if early files are missing, so pass
  `first_image_number` when it is known.

[`src/ffs/`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/src/ffs
[`spotfinder/`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/spotfinder
