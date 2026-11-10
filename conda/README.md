# Conda environments

`development/environment.yml` is the only file here that carries
versions. Every other environment is derived from it by `derive.py`,
which keeps or drops the packages named in a list file and writes the
result beside that list, so a pin moves everywhere at once and the
environments cannot drift apart. The derived `environment.yml` files are
ignored by git; edit the development file, never those.

The layout exists so that Dependabot can keep the pins current. It only
reads files named `environment.yml` and opens a separate pull request
for each one, so with every version in a single file each update is one
pull request that moves all the environments together.

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

To add a package, pin it in `development/environment.yml`, and add its
name to `runtime/packages` if the service needs it at run time.
`derive.py` fails on a listed name that the development file does not
contain, so a typo or an upstream rename shows up at build time.

To see what a derived environment contains:

```bash
python conda/derive.py --keep conda/runtime/packages
cat conda/runtime/environment.yml
```
