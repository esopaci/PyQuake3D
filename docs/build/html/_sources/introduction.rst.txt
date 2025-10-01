Introduction
============

PyQuake3D is an open-source Python framework designed to simulate 3D earthquake rupture
dynamics with realistic fault geometries and rate-and-state friction laws. The goal of
PyQuake3D is to provide a high-performance, extensible tool for the earthquake physics
community that bridges simple benchmark problems with full-scale, physics-rich scenarios.

The code supports multiple execution backends:

* **Single CPU/GPU backend** — constructs dense stiffness matrices using direct evaluation
  of all source–receiver interactions. GPU acceleration is supported via `CuPy` and
  parallelism through Python’s ``ProcessPoolExecutor``.
* **MPI-based CPU backend** — implements a memory-efficient H-matrix representation of the
  stiffness matrix, distributed across processors via ``mpi4py``. This is the recommended
  path for large models (>40,000 elements) and HPC systems.

With this design, PyQuake3D scales from exploratory models on laptops to high-resolution
simulations on clusters. The modular structure makes it straightforward to extend the
framework with new rheologies, boundary conditions, or couplings to geodynamic models.

PyQuake3D is developed and maintained by Rongjiang Tang and Luca Dal Zilio. 
The source code, documentation, and tutorials are hosted on GitHub:

    https://github.com/Computational-Geophysics/PyQuake3D