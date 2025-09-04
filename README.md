<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/logo/PyQuake3D_grey.png" alt="PyQuake3D logo" style="width:100%;">
</p>

<h1 align="center">PyQuake3D</h1>
<h3 align="center">A Python tool for 3-D earthquake sequence simulations of seismic and aseismic slip</h3>

**PyQuake3D** is a high-performance Python-based Boundary Element Method (BEM) code for simulating sequences of seismic and aseismic slip (SEAS) on a complex 3D fault geometry governed by rate- and state-dependent friction. It combines physics-based modeling with modern parallel computing tools (MPI, GPU acceleration via CuPy) to solve a variety of earthquake cycle and rupture problems. This document provides an overview of how to use the script, as well as a detailed description of the input parameters.

## Authors and Contact

**PyQuake3D** was developed by [Dr. Rongjiang Tang](https://scholar.google.com/citations?user=_4cR3zMAAAAJ&hl=zh-CN) and [Dr. Luca Dal Zilio](https://www.lucadalzilio.net/).  
We welcome contributions to the project—please follow the contribution guidelines and help us maintain a clean, consistent codebase.

For questions, suggestions, or collaboration opportunities, feel free to reach out:

-  rongjiang@csj.uestc.edu.cn  
-  luca.dalzilio@ntu.edu.sg

Please refer to the [Code Manual (PDF)](PyQuake3D_User_Manual.pdf)

<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/framework/turkey_displacement.png" alt="Turkey Displacement">
</p>

## Features

-  3D non-planar quasi-dynamic earthquake cycle simulations
-  Support for rate-and-state aging friction laws
-  Support for Hierarchical matrix storage and calculation
-  Support for GPU acceleration via CuPy
-  MPI acceleration support 
-  Suitable for large model earthquake cycle simulation
-  Support for pore fluid pressure varing with slip due to inelastic processes including dilatancy, pore compaction.

<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/framework/framework.png" alt="Framework Overview">
</p>

## Quick Start
A step by step tutorial on how to install and run [BP5-QD_low_resolution case](tutorials/BP5-QD_low_resolution/tutorial_BP5.ipynb) and [circular_asperity_low_resolution case](tutorials/circular_asperity_low_resolution/tutorial_circular_asperity.ipynb). The former uses  parameters to set the initial model, while the latter uses external files to import the initial model.

## Installation

### Python Requirements

PyQuake3D supports Python 3.8 and above, so there is no need to specify any version when installing the dependent libraries.

- `numpy`
- `matplotlib`
- `scipy`
- `joblib`
- `mpi4py`
- `pyvista`
- `imageio`
- `psutil`
- `h5py`

  Use pip for the quick installation:
```bash
pip install -r requirements.txt
```
Or use conda to install:
```bash
conda env update -f environment.yml
```
Install cupy if you want to use GPU acceleration, we recommened to use conda (e.g. CUDA 11.8):conda install -c conda-forge cupy cudatoolkit=11.8

### C++ Requirements

The TDstressFS_C.cpp in folder src is a C++ source file that computes Green's functions, translated from the Python script TDstressFS.py to leverage C++'s performance for efficient numerical calculations. It is compiled into a dynamic library, TDstressFS_C.so, using a provided Makefile, which must be executed with the make command before running the code to ensure compatibility across different computing environments. The generated library is called by the Python script Hmatrix.py via dynamic loading (e.g., using ctypes). To use it, navigate to the code directory src, run make to build TDstressFS_C.so.

## Running the Script
PyQuake3D provides two versions of the code, GPU and MPI, which can be run using different main functions：main_gpu or main_mpi. main_mpi uses Hmatrix to reduce memory overhead and thus is more suitable for larger models with more than 40,000 cells.
## For single GPU/CPU version, use the following command:
python -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```bash
python src/main_gpu.py -g examples/BP5-QD/bp5t.msh -p examples/BP5-QD/parameter.txt
```
Ensure you modify the input parameter (`parameter.txt`) as follows:
- `InputHetoparamter`: `True`
- `Inputparamter file`: `bp5tparam.dat`

## For MPI version, use the following command:
To run the PyQuake3D MPI script, use the following command at root directory:
```bash
mpirun -np 10 python main.py -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```
Where 10 is the number of virtual cpus. Note that using the mpiexec instead in Windows environment.

For example:
```
To execute benchmarks like BP5-QD, use:
```bash
In the PyQuake3D root directory, To run the BP5-QD benchmark:
mpirun -np 10 python src/main_gpu.py -g examples/BP5-QD/bp5t.msh -p examples/BP5-QD/parameter.txt


To run the HF-model:
mpirun -np 10 python src/main_mpi.py -g examples/HF-model/HFmodel.msh -p examples/HF-model/parameter.txt

To run the  EAFZ-model:
mpirun -np 10 python src/main_mpi.py -g examples/EAFZ-model/turkey.msh -p examples/EAFZ-model/parameter.txt

To run the  Lab-model:
mpirun -np 10 python src/main_mpi.py -g examples/Lab-model/lab.msh -p examples/Lab-model/parameter.txt
```


## Parameters Setting
The simulation parameters are implemented by modifying the parameter.txt file, rather than by changing the source code. The heterogeneous stress and friction parameters are imported from external files. Please refer to [Code Manual (PDF)](PyQuake3D_User_Manual.pdf) for description of parameter details. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
We acknowledge the support and feedback provided by the broader scientific community and all contributors to this work.

Development of the Python-based BEM algorithm was informed by the HBI code introduced in:  
**Ozawa, S., Ida, A., Hoshino, T., & Ando, R.** (2023). *Large-scale earthquake sequence simulations on 3-D non-planar faults using the boundary element method accelerated by lattice H-matrices*. **Geophysical Journal International**, 232(3), 1471–1481.  
[https://doi.org/10.1093/gji/ggad042](https://doi.org/10.1093/gji/ggad042)

The implementation of the stress Green’s functions builds on MATLAB routines from:  
**Nikkhoo, M., & Walter, T. R.** (2015). *Triangular dislocation: an analytical, artefact-free solution*. **Geophysical Journal International**, 201(2), 1119–1141.  
[https://doi.org/10.1093/gji/ggv035](https://doi.org/10.1093/gji/ggv035)

We sincerely thank Ryosuke Ando and So Ozawa for their valuable guidance in the development of the code. We also thank Steffen Börm for his assistance with H-matrix implementation and T. Ben Thompson for his assistance with the H-matrix compression via Adaptive Cross Approximation (ACA).

## Examples

Explore selected simulations performed with PyQuake3D:

- [*Seismic cycles on a planar fault with frictional heterogeneity*](https://www.youtube.com/watch?v=N_yA4uY77C0)
- [*Seismic cycles on the East Anatolian Fault Zone*](https://www.youtube.com/watch?v=oFy3FSLs3UQ)



