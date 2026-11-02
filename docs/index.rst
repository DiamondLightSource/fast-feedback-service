fast-feedback-service documentation
===================================

Version |release|

``fast-feedback-service`` processes X-ray diffraction data with GPU
acceleration. It serves both as a real-time feedback service on the
beamline, where GPU spotfinding drives X-ray centring, and as an
accelerated spotfind-index-integrate pipeline for fast feedback on
complete datasets.

Spotfinding and integration run on the GPU. Rotation data is indexed
by the C++ baseline indexer; stills and SSX data are indexed on the
GPU by `ffbidx <https://github.com/paulscherrerinstitute/fast-feedback-indexer>`_.

The API reference below covers this repository's C++ and CUDA code
and is generated from its Doxygen comments. The Python service that
drives the compiled programs is not documented here yet.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   readme

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/library_root

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
