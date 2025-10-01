Versions
=========

v1.0.0 (2025-09-30)
-------------------

We are pleased to announce the first public release of **PyQuake3D** – a Python-based simulation framework for modeling 3-D earthquake sequences, capturing both seismic and aseismic fault slip.

Key Features
~~~~~~~~~~~~

- Simulation of multi-cycle earthquake behavior on 3-D faults
- Support for both seismic rupture and aseismic creep
- GPU acceleration using **CuPy**
- MPI parallelization for large-scale computations
- Modular finite-difference / finite-volume numerical implementation
- Geometry-based input using ``.msh`` files
- Parameter-driven simulation control (``parameter.txt``)
- Diagnostic output and visualization tools via HDF5 and Matplotlib
- Includes benchmark examples such as **SCEC BP5-QD**

Example Simulations
~~~~~~~~~~~~~~~~~~~

- `Planar Fault with Frictional Heterogeneity <https://www.youtube.com/watch?v=N_yA4uY77C0>`_
- `East Anatolian Fault Zone Model <https://www.youtube.com/watch?v=oFy3FSLs3UQ>`_
