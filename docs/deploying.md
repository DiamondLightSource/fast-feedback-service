# Deploying to the module system

[`deploy.sh`] builds the service into a conda prefix and publishes a
modulefile pointing at it. It targets Diamond's install tree under
`/dls_sw` and its module system; elsewhere, run the container or
[build from source](building.md).

It follows the same two-environment split as the [`Dockerfile`]:
[`conda/development/environment.yml`] creates the build environment, and
the one that ships is derived from it with only the packages listed in
`conda/runtime/packages`. Nothing compiles into the deployed tree, so no
compiler, cmake or test framework ends up on a user's PATH.

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
                           the module, and must equal the version the
                           source resolves to. dev takes any version.
                           Releases are YYYY.MM, without the month's
                           leading zero: v2027.01 resolves to 2027.1.
                           (default: dev)
    -p, --prefix PATH      Install prefix, overriding
                           <install-root>/<version>
        --install-root DIR Base of the install tree (default: /dls_sw/apps/fast-feedback-service)
        --module-root DIR  Where modulefiles live (default: /dls_sw/apps/Modules/modulefiles/fast-feedback-service)
        --module-name NAME Module name, overriding <version>
    -b, --build-env PATH   Build environment prefix (default: /tmp/ffs-build-env)
    -c, --cuda MODULE      CUDA module to build against (default: cuda/<the
                           Dockerfile's CUDA_VERSION>)
    -j, --jobs N           Cap parallel build jobs
    -r, --recreate         Delete and recreate both conda environments
    -i, --incremental      Reuse the build directory instead of starting
                           clean. Faster while iterating, but a cached
                           path from an earlier prefix will be believed.
    -m, --modulefile       Also install the modulefile
    -l, --latest           Point the latest symlink at this module.
                           Requires --modulefile.
    -h, --help             Show this help
```

## Naming

`--version` sets both the install directory and the module name, so
`--version 2026.11` installs to `<install-root>/2026.11` and publishes
`fast-feedback-service/2026.11`.

A release name is checked against the version setuptools_scm resolves
from the source before anything is installed, and again against the
installed package afterwards. `--version 2026.11` therefore only deploys
from a clean checkout of the `v2026.11` tag; anything else stops with the
version the source actually resolves to. `dev` is the rolling name and
takes whatever the source resolves to, which the modulefile's
`module-whatis` then records.

The other path options override the derived values individually, which
is how to stage a build somewhere private before publishing:

```bash
./module/deploy.sh --prefix /scratch/me/ffs \
                   --module-root /scratch/me/modules --modulefile
```

The modulefile is only written with `--modulefile`, since that is the
step that makes a build visible to everyone on the machine.

## The latest pointer

`--latest` points `fast-feedback-service/latest` at the module just
published, so a recipe can say

```bash
module load fast-feedback-service/latest
```

and follow releases without being edited. It is a symlink in the module
directory beside the versions, which is how the rest of `/dls_sw/apps`
does it, and it is a separate flag from `--modulefile` because it
changes what existing recipes resolve to rather than adding something
new alongside them.

The symlinked file carries the concrete version in its `module-whatis`,
so `module whatis fast-feedback-service/latest` names the release that
is actually loaded rather than the word latest. Nothing sorts version
names: latest means the last one deployed with the flag, so publishing
an older release with `--latest` moves the pointer backwards.

## Releasing a version

A release has to deploy from a clean checkout of its tag. Any checkout
will do; the suggested method is a worktree beside the main checkout:

```bash
git fetch origin
git tag -a v2026.11 -m "fast-feedback-service 2026.11" origin/main
git worktree add ../ffs-2026.11 v2026.11
git -C ../ffs-2026.11 submodule update --init --recursive
../ffs-2026.11/module/deploy.sh --version 2026.11 --modulefile --latest
git push origin v2026.11
git worktree remove --force ../ffs-2026.11
```

`git worktree add` leaves submodules empty, and the build needs dx2.
The tag is pushed once the module works, since pushing it publishes the
container image. The installed prefix does not depend on the worktree;
removing it needs `--force` because it holds a submodule and the
untracked build directory.

## The modulefile

[`module/modulefile`] is a template. `deploy.sh` substitutes the `@...@`
placeholders and refuses to publish if any survive.

It sets PATH and `LD_LIBRARY_PATH`, requires the CUDA module it was
built against, and declares `conflict` so two versions cannot be loaded
at once.

It deliberately does not set `SPOTFINDER`, `INDEXER` or `INTEGRATOR`.
`find_executable` prefers those variables over PATH so that a local
build can be used without reordering PATH, and a module that sets them
takes that away. The container sets them because it has no module
system to lean on.

[`deploy.sh`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/module/deploy.sh
[`Dockerfile`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/Dockerfile
[`conda/development/environment.yml`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/conda/development/environment.yml
[`module/modulefile`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/module/modulefile
