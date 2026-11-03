/**
 * @file version.hpp
 * @brief The project version, as resolved by setuptools_scm.
 *
 * The string is generated into version.cc by cmake/Modules/WriteVersion.cmake
 * and compiled into the ffs_version library. It is the same string the python
 * package carries as ffs.__version__, so a program and the package it was
 * built with always report the same version.
 *
 * Only the programs and ffs_cli, the argument parsing and banner they share,
 * link ffs_version. In a plain cmake tree the version is resolved again on
 * every build and changes with each commit, so anything linking it relinks
 * then; ffs_common, which the tests and the python extensions link, stays
 * independent of it.
 */
#pragma once

#include <string_view>

/**
 * @brief The PEP 440 version string, e.g. "0.2.dev3+gabc1234".
 *
 * Not in a namespace named ffs: glibc declares ffs() in the global namespace,
 * and a namespace of the same name cannot coexist with it.
 */
std::string_view ffs_version();
