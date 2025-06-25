

import numpy as np
import sys
import matplotlib.pyplot as plt

from math import *
import time
import argparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import pyvista as pv

# 读取vtk文件
mesh = pv.read("step0.vtk")

# 查看网格信息
print(mesh)

# 访问点坐标
points = mesh.points


# 访问单元信息（cells）
cells = mesh.cells


for name, array in mesh.cell_data.items():
    print(f"{name}: shape={array.shape}, dtype={array.dtype}")


Ncell=cells.shape[0]
rake=mesh.cell_data['rake[Degree]']
a=mesh.cell_data['a']
b=mesh.cell_data['b']
dc=mesh.cell_data['dc']
#f0=mesh.cell_data['rake[Degree]']
Tt=mesh.cell_data['Shear_[MPa]']
Tno=mesh.cell_data['Normal_[MPa]']
slipv=mesh.cell_data['Slipv[m/s]']

#shear_loading=values[:Ncell,8]
#normal_loading=values[:Ncell,9]
print(rake.shape,cells.shape)

f=open('initcondion.txt','w')
for i in range(len(rake)):
    f.write('%f %f %f %f 0.6 %f %f %.20f 0 0\n'%(rake[i],a[i],b[i],dc[i],Tt[i],Tno[i],slipv[i]))
f.close()


