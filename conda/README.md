# Conda environments

`development/environment.yml` is the only file here that carries
versions. Every other environment is derived from it by `derive.py`,
which keeps or drops the packages named in a list file and writes the
result beside that list, so a pin moves everywhere at once and the
environments cannot drift apart. The derived `environment.yml` files are
ignored by git; edit the development file, never those.

- `runtime/packages` keeps what the deployed service ships. The
  `Dockerfile` and `module/deploy.sh` derive the runtime environment
  from it.
- `macos/packages` drops the Linux compilers for the macOS test job,
  where the compilers metapackage supplies clang instead.

The runtime list includes `scikit-build-core`, `setuptools_scm` and
`nanobind`, although nothing uses them at run time. pip installs the
extension into the runtime environment with `--no-build-isolation`, so
the build backend and nanobind's cmake config come from that
interpreter.

`update.py` keeps the pins current. Once a month the Conda update
workflow runs it to move every pin to the newest version that solves
together on linux-64, linux-aarch64 and osx-arm64, and opens one pull
request with the result. Run by hand, it rewrites the development file
in place and prints each change.

To add a package, pin it in `development/environment.yml`, and add its
name to `runtime/packages` if the service needs it at run time.
`derive.py` fails on a listed name that the development file does not
contain, so a typo or an upstream rename shows up at build time.

On macOS the development file cannot install, as its compilers are
Linux only; derive the macOS environment and create that instead:

```bash
python conda/derive.py --drop conda/macos/packages
mamba env create -p ./ENV -f conda/macos/environment.yml
```

The runtime environment is made the same way, to reproduce what the
image ships:

```bash
python conda/derive.py --keep conda/runtime/packages
mamba env create -p ./ENV-runtime -f conda/runtime/environment.yml
```

To update the pins locally, with micromamba or mamba on `PATH`:

```bash
python conda/update.py
git diff conda/development/environment.yml
```
