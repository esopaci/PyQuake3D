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

Please refer to the [Code Manual (PDF)](https://github.com/Computational-Geophysics/PyQuake3D/blob/main/user_manual/user_manual.pdf)

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

<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/framework/framework.png" alt="Framework Overview">
</p>

## Installation

### Python Requirements

PyQuake3D requires Python ≥ 3.8 and the following libraries:

- `numpy >= 1.20`
- `cupy == 10.6.0`
- `matplotlib == 3.2.2`
- `scipy == 1.10.1`
- `joblib == 0.16.0`
- `mpi4py == 4.0.0`
- `ctypes == 1.1.0`
- `pyvista == 0.45.2`

  Use pip for the quick installation:
```bash
pip install -r dependences.txt
```

## Running the Script

To run the PyQuake3D MPI script, use the following command:
```bash
mpirun -np 10 python -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```
Where 10 is the number of virtual cpus. Note that using the mpiexec instead in Windows environment.

For example:
```
To execute benchmarks like BP5-QD, use:
```bash
In the PyQuake3D_MPI_master directory, To run the BP5-QD benchmark:
mpirun -np 10 python src/main.py -g examples/BP5-QD/bp5t.msh -p examples/BP5-QD/parameter.txt
Ensure you modify the input parameter (`parameter.txt`) as follows:
- `Corefunc directory`: `bp5t_core`
- `InputHetoparamter`: `True`
- `Inputparamter file`: `bp5tparam.dat`

To run the HF-model:
mpirun -np 10 python src/main.py -g examples/HF-model/HFmodel.msh -p examples/HF-model/parameter.txt

To run the  EAFZ-model:
mpirun -np 10 python src/main.py -g examples/EAFZ-model/turkey.msh -p examples/EAFZ-model/parameter.txt

To run the  Lab-model:
mpirun -np 10 python src/main.py -g examples/Lab-model/lab.msh -p examples/Lab-model/parameter.txt
```


## Parameters Setting
The simulation parameters are implemented by modifying the parameter.txt file, rather than by changing the source code. The heterogeneous stress and friction parameters are imported from external files. 
## General Parameters setting
| Parameter                  | Default                   | Description                                                                                                            |
|----------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------------|
| `Corefunc directory`       |                           | The storage path for the kernel function matrix composed of stress Green's functions                                   |
| `Hmatrix_mpi_plot`         | False                     | If `True`, draw the Hmatirx structure diagram, with different colors representing the sub-matrices calculated by different processes. Only availible for MPI verison. |
| `Node_order`               | False                     | If `True`, the node order of the triangular element is clockwise                                                      |
| `save Corefunc`            | False                     | If `True`, save corefuns Save the kernel function so that it does not need to be recalculated for the next time       |
| `Scale_km`                 | True                      | If `True`, the coordinates will be scaled up by a factor of 1000, meaning they are modeled in kilometers, which is applicable to natural earthquakes; otherwise, the coordinates remain unchanged, meaning they are modeled in meters, which is applicable to laboratory earthquakes. |
| `Input Hetoparamter`       | False                     | If `True`, the heterogeneous stress and friction parameters are imported from external files.                         |
| `Inputparamter file`       |                           | The file name of imported heterogeneous stress and friction parameters                                                |
| `Processors`               | 50                        | The number of processors in ProcessPoolExecutor to parallelize Green's function calculations. Only availible for GPU verison. |
| `Batch_size`               | 1000                      | The number of batches in ProcessPoolExecutor to parallelize Green's function calculations. Only availible for GPU verison. |
| `Lame constants`           | $0.32 \times 10^{11}$ Pa | The first Lame constant                                                                                                |
| `Shear modulus`            | $0.32 \times 10^{11}$ Pa | Shear modulus                                                                                                          |
| `Rock density`             | $2670\ \text{kg/m}^3$     | Rock mass density                                                                                                      |
| `Reference slip rate`      | $1 \times 10^{-6}$        | Reference slip rate                                                                                                    |
| `Reference friction coefficient` | 0.6                        | Reference friction coefficient                                                                                        |
| `Plate loading rate`       | $1 \times 10^{-6}$        | Plate loading rate                                                                                                    |
                                                                                                                                                                                                        |


## Stress and Frition Settings
| Parameter                                  | Default   | Description                                                                                         |
|--------------------------------------------|-----------|-------------------------------------------------------------------------------------------------|
| `Half space`                              | False     | If 'True', calculating half-space green's functions                                              |
| `Fix_Tn`                                  | True      | If 'True',  fixed the normal stress                                                             |
| `Vertical principal stress (ssv)`         | 1.0       | The vertical principal stress scale: the real vertical principal stress is obtained by multiplying the scale and the value |
| `Maximum horizontal principal stress (ssh1)` | 1.6       | Maximum horizontal principal stress scale.                                                      |
| `Minimum horizontal principal stress(ssh2)` | 0.6       | Minimum horizontal principal stress scale                                                       |
| `Angle between ssh1 and X-axis`           | 30°       | Angle between maximum horizontal principal stress and X-axis.                                    |
| `Vertical principal stress value`          | 50 MPa    | Vertical principal stress value                                                                  |
| `Vertical principal stress value varies with depth` | True      | If True, Vertical principal stress value varies with depth                                      |
| `Vertical principal stress value varies with depth` | True      | If vertical principal stress value, it maintains a constant value at the conversion depth, and the horizontal principal stress value also changes with depth simultaneously |
| Turnning depth                            | 5000 m    | If Vertical principal stress value varies with depth is true, starting at this depth, the stress no longer changes with depth |
| `Shear traction solved from stress tensor` | False     | If 'True', the non-uniform shear stress is projected onto the curved fault surface by the stress tensor |
| `Rake solved from stress tensor`           | False     | If 'True', the non-uniform rakes are solved from the stress tensor.                              |
| `Fix_rake`                                | 30°       | If 'True', Set fixed rakes if 'Rake solved from stress tensor' is 'False'.                      |
| `Widths of VS region`                      | 5000 m    | The width of the velocity weakening region.                                                     |
| `Widths of surface VS region`              | 2000 m    | Widths of surface VS region                                                                      |
| `Transition region  from VS to VW region` | 3000 m    | Transition region width from VS to VW region                                                    |
                                                                                                                                                              |



## Nucleation and Friction Setting
| Parameter                                           | Default   | Description                                                                                   |
|-----------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------|
| `Set_nucleation`                                    | False     | If True, sets a patch whose shear stress and sliding rate are significantly greater than the surrounding area to meet the nucleation requirements. |
| `Radius of nucleation`                              | 8000 m    | The radius of the nucleation region                                                           |
| `Nuclea_posx`                                       | 34000 m   | Posx of Nucleation                                                                            |
| `Nuclea_posy`                                       | 15000 m   | Posy of Nucleation                                                                            |
| `Nuclea_posz`                                       | -15000 m  | Posz of Nucleation                                                                            |
| `Rate-and-state parameters a in VS region`          | 0.04      | Rate-and-state parameters a in VS region                                                      |
| `Rate-and-state parameters b in VS region`          | 0.03      | Rate-and-state parameters a in VS region                                                      |
| `Characteristic slip distance in VS region`         | 0.13 m    | Characteristic slip distance in VS region                                                     |
| `Rate-and-state parameters a in VW region`          | 0.004     | Rate-and-state parameters a in VW region                                                      |
| `Rate-and-state parameters a in VW region`          | 0.03      | Rate-and-state parameters a in VW region                                                      |
| `Characteristic slip distance in VW region`         | 0.13 m    | Characteristic slip distance in VW regioN                                                     |
| `Rate-and-state parameters a in nucleation region`  | 0.004     | Rate-and-state parameters a in nucleation region                                              |
| `Rate-and-state parameters a in nucleation region`  | 0.03      | Rate-and-state parameters a in nucleation region                                              |
| `Characteristic slip distance in nucleation  region`| 0.14 m    | Characteristic slip distance in nucleation  regioN                                            |
| `Initial slip rate in nucleation region`            | 3e-2      | Initial slip rate in nucleation region                                                        |
| `ChangefriA`                                        | False     | If True, a changes gradually b remains unchanged, vice versa                                  |
| `Initlab`                                           | True      | If True, setting random non-uniform normal stress                                             |



## Output Setting
| Parameter           | Default | Description                                                             |
|---------------------|---------|-------------------------------------------------------------------------|
| `totaloutputsteps`  | 2000    | The number of calculating time steps.                                   |
| `outsteps`          | 50      | The time step interval for outputting the VTK files.                    |
| `outputSLIPV`       | False   | If True, output slip rate for each step.                                |
| `outputTt`          | False   | If True, output shear stress for each step.                              |
| `outputstv`         | True    | If True, the VTK files will be saved in out directroy.                  |
| `outputmatrix`      | False   | If True, the matrix format txt files will be saved in out directroy.    |

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



