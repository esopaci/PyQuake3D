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


if __name__ == "__main__":

	arg_list = sys.argv
	
	simpath = arg_list[1]
	
	p = Ptool(simpath)
	
	p.Vdyn = 1e-2
	
	if 'ts' in arg_list[1:]:
		# time series plot
		p.plot_timeseries()
	if 'a2' in arg_list[1:]:
		# Animation in 2D
		p.animation2D(N_interval=10)
	if 'a3' in arg_list[1:]:
		# ANIMATION IN 3D DOMAIN
		p.animation3D(N_interval=1,azim=80)
	if 'event' in arg_list[1:]:
		# Generating event file
		p.extract_slip_info()
