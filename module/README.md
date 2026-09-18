# Deploying to the module system

`deploy.sh` builds the service into a conda prefix and publishes a
modulefile pointing at it.

It follows the same two-environment split as the `Dockerfile`:
`environment.yml` creates the build environment and
`runtime-environment.yml` creates the one that ships. Nothing compiles
into the deployed tree, so no compiler, cmake or test framework ends up
on a user's PATH.

## Quick start

```bash
./module/deploy.sh                  # build into <install-root>/dev
./module/deploy.sh --modulefile     # ... and publish the module
./module/deploy.sh --help
```

**Options:**
```bash
./module/deploy.sh [OPTIONS]

OPTIONS:
    -v, --version NAME     Release name; names the install directory and
                           the module (default: dev)
    -p, --prefix PATH      Install prefix, overriding
                           <install-root>/<version>
        --install-root DIR Base of the install tree (default: /dls_sw/apps/fast-feedback-service)
        --module-root DIR  Where modulefiles live (default: /dls_sw/apps/Modules/modulefiles/fast-feedback-service)
        --module-name NAME Module name, overriding <version>
    -b, --build-env PATH   Build environment prefix (default: /tmp/ffs-build-env)
    -c, --cuda MODULE      CUDA module to build against (default: cuda/13.0.2)
    -j, --jobs N           Cap parallel build jobs
    -r, --recreate         Delete and recreate both conda environments
    -i, --incremental      Reuse the build directory instead of starting
                           clean. Faster while iterating, but a cached
                           path from an earlier prefix will be believed.
    -m, --modulefile       Also install the modulefile
    -h, --help             Show this help
```

## Naming

`--version` sets both the install directory and the module name, so
`--version 1.0.0` installs to `<install-root>/1.0.0` and publishes
`fast-feedback-service/1.0.0`.

The other path options override the derived values individually, which
is how to stage a build somewhere private before publishing:

```bash
./module/deploy.sh --prefix /scratch/me/ffs \
                   --module-root /scratch/me/modules --modulefile
```

The modulefile is only written with `--modulefile`, since that is the
step that makes a build visible to everyone on the machine.

## The modulefile

`module/modulefile` is a template. `deploy.sh` substitutes the `@...@`
placeholders and refuses to publish if any survive.

It sets PATH and `LD_LIBRARY_PATH`, requires the CUDA module it was
built against, and declares `conflict` so two versions cannot be loaded
at once.

It deliberately does not set `SPOTFINDER`, `INDEXER` or `INTEGRATOR`.
`find_executable` prefers those variables over PATH so that a local
build can be used without reordering PATH, and a module that sets them
takes that away. The container sets them because it has no module
system to lean on.
