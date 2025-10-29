#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:31:27 2025
This is an example script using Plot_tools 
@author: eyup
"""

import os 
import sys
import pyvista as pv
import numpy as np

PyQuake_path = f'{os.path.expanduser("~")}/PyQuake3D'

sys.path.append(os.path.join(PyQuake_path, 'utils'))

from Plot_tool import Ptool

simpath = '/Users/eyup/workspace/mong_3D/fast/L60'

# simpath = sys.argv[1]
p = Ptool(simpath)
p.Vdyn = 1e-2
# time series plot
# p.plot_timeseries()

# Animation in 2D
# p.animation2D(N_interval=10)

# ANIMATION IN 3D DOMAIN
# p.animation3D(N_interval=10,azim=80)

# Generating event file
p.extract_slip_info()
