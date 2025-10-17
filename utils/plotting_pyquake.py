#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:16:59 2025
This script helps plotting PyQuake3D outputs
@author: eyup
"""

import pyvista as pv
import pandas as pd 
import os 
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.tri as tri


class Ptool:
    
    t_yr = 365*3600*24
    field = ['Normal_[MPa]', 
              'Pore_pressure[MPa]', 
              'Shear_[MPa]', 
              'Shear_1[MPa]', 
              'Shear_2[MPa]', 
              'rake[Degree]', 
              'state', 
              'Slipv[m/s]', 
              'Slipv1[m/s]', 
              'Slipv2[m/s]', 
              'a', 
              'b', 
              'a-b', 
              'dc', 
              'fric', 
              'slip', 
              'slip1', 
              'slip2', 
              'slip_plate']
    

    def __init__(self, path):
    
        self.path = path
        
        
        ##  Get snapshots
        self.out_folder = os.path.join(path, 'out_vtk')
        self.vtk_files = [int(x.split('.vtu')[0][4:]) for x in os.listdir(self.out_folder) if x.endswith('.vtu') ]
        self.steps = np.sort(self.vtk_files)
        self.state_file = os.path.join(path, 'state.txt')
        self.N_steps= len(self.vtk_files)
    
    def read_statefile(self):
        
        vmax = pd.read_csv(
            self.state_file,
            sep = '\s+', skiprows=15,low_memory=True,
            names=['Iteration', 'dt', 'slipv1', 'slipv2', 'time(s)', 'time(h)']
            )
        vmax = vmax.dropna().astype(float)
        return vmax
    
    def plot_timeseries(self):
        fig,ax = plt.subplots(1,1, figsize = (10,6), clear=True)
        
        vmax = self.read_statefile()

        ax.set_xlabel('time [yr]')
        ax.set_ylabel('log($V_{max}$) [m/s]')
        ax.semilogy(vmax['time(s)']/self.t_yr, 
                    vmax['slipv1'])
        
        fig.savefig(os.path.join(sys.path,'max_time_series.jpg'), dpi = 300, bbox_inches='tight')
        
    def animation2D(self, vmin = -9, vmax = 0):
        
        # --- Plot with Matplotlib ---
        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot(111, projection="3d")
        fig,(ax,ax1) = plt.subplots(2,1, figsize = (8,6))
        ax.set_xlabel('X[km]')
        ax.set_ylabel('Z[km]')

        
        
        # ax.view_init(elev=30, azim=10, roll=0) 
        # ax.set_box_aspect([0.5, 4, 0.75]) 
        # ax1 = fig.add_subplot(422)
        # plt.tight_layout()
        df = self.read_statefile()

        
        ax1.set_xlabel('time [yr]')
        ax1.set_ylabel('V [m/s]')
        ax1.semilogy(df['time(s)']/self.t_yr, 
                    df['slipv1'], lw = 1)
        
        line, = ax1.semilogy(df['time(s)'].iloc[0]/self.t_yr, 
                    df['maximum_slip_rate(m/s)'].iloc[0], color = 'r', marker = 'o')
        
        timetext = ax1.text(0.0,1.0, "Y{:0>5.0f} D{:0>3.0f}-{:0>2.0f}:{:0>2.0f}:{:0>2.0f}".format(0,
                                                  0,
                                                  0,
                                                  0,
                                                  0),
                           horizontalalignment='left',
                            verticalalignment='bottom',
                            transform = ax.transAxes)

        
        mesh = pv.read(os.path.join(self.out_folder, f'step{self.steps[0]}.vtk'))

        


        V = mesh.cell_data['Slipv[m/s]']
        cells = mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
        triangles = cells[:, 1:]         # drop the "3"
        points = mesh.points[triangles]  # shape (n_cells, 3, 3)
        
        triang = tri.Triangulation(points[:,0,1], points[:,0,2])

        
        tpc = ax.tripcolor(triang, V, cmap='viridis', 
                          norm=LogNorm(vmin=10**vmin, vmax=10**vmax), 
                          )
        
        cbar = fig.colorbar(tpc, ax=ax, label='V [m/s]')

        def update(i):
            
            step = self.steps[i]
            print(step)
            mesh = pv.read(os.path.join(self.out_folder, f'step{step}.vtu'))
            V = mesh.cell_data['slipv']
            # sctr.set_offsets(np.c_[x_data[:frame+1], y_data[:frame+1]])
            tpc.set_array(V)
            # surf.set_norm(LogNorm(vmin=10**vmin, vmax=10**vmax))
                        
            time = df.iloc[int(step),-2]
            sliprate = df.iloc[int(step),2]
            
            line.set_data([time/self.t_yr], [sliprate])

                        
            timetext.set_text("Y{:0>5.0f} D{:0>3.0f} - {:0>2.0f}:{:0>2.0f}:{:0>4.2f}".format( time/(365*3600*24),
                                                  (time/3600/24)%(365),
                                                  (time/3600)%24,
                                                  (time/60)%60,
                                                  time%60)
                        )

            return tpc, timetext, line
        
        anim = FuncAnimation(fig, update, frames=np.arange(1,self.N_steps,1), blit=True)
        
        anim.save(os.path.join(self.path,"animation1.mp4"), fps=10, dpi=150)        
        
        
    def animation3D(self, vmin = -10, vmax = 0, azim = 30, interval = 2):
        
        # --- Plot with Matplotlib ---
        fig = plt.figure(figsize=(10,8))
        
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.05, 0.35, 0.9, 0.7])
        ax.view_init(elev=10, azim=azim) 
        ax.set_box_aspect([1.6, 0.8, 0.6]) 
        ax1 = fig.add_subplot(611)
        ax1.set_position([0.1, 0.05, 0.85, 0.2])
        
        df = self.read_statefile()

        
        ax1.set_xlabel('time [yr]')
        ax1.set_ylabel('V [m/s]')
        ax1.semilogy(df['time(s)']/self.t_yr, 
                    df['slipv1'], lw = 1)
        
        line, = ax1.semilogy(df['time(s)'].iloc[0]/self.t_yr, 
                    df['slipv1'].iloc[0], color = 'r', marker = 'o')
        
        timetext = ax1.text(0.0,0.8, "Y{:0>5.0f} D{:0>3.0f}-{:0>2.0f}:{:0>2.0f}:{:0>2.0f}".format(0,
                                                  0,
                                                  0,
                                                  0,
                                                  0),
                           horizontalalignment='left',
                            verticalalignment='bottom',
                            transform = ax.transAxes)


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        mesh = pv.read(os.path.join(self.out_folder, f'step{self.steps[0]}.vtu'))
        # mesh = pv.read(os.path.join(p.out_folder, 'step100.vtk'))

        # Extract vertex coordinates
        points = mesh.points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        tri_cells = mesh.extract_cells(np.where(mesh.celltypes == pv.CellType.TRIANGLE)[0])
        triangles = tri_cells.cells.reshape(-1, 4)[:, 1:4]  # skip the first '3' entry
        
        
        V = mesh.cell_data['Slipv[m/s]']
        
        surf = ax.plot_trisurf(
            x*1E-3, y*1E-3, z*1E-3,
            triangles=triangles,
            cmap="magma",
            linewidth=0.05,
            edgecolor="none",
            alpha=1.0
        )
        
        
        surf.set_array(V)
        surf.set_norm(LogNorm(vmin=10**vmin, vmax=10**vmax))
        fig.colorbar(surf, ax=ax, shrink=0.3, label="V[m/s]", 
                     orientation = 'vertical',
                     location='right', pad = 0.1,
                     extend='both',
                     norm=LogNorm(vmin=10**vmin, vmax=10**vmax)
                     )
        
        
        
        def update(i):
            # try:
            step = int(self.steps[i])
            print(step)
            mesh = pv.read(os.path.join(self.out_folder, f'step{step}.vtu'))
            V = mesh.cell_data['Slipv[m/s]']

            surf.set_array(V)
            surf.set_norm(LogNorm(vmin=10**vmin, vmax=10**vmax, clip=True))
            
            temp = df[df.Iteration==int(self.steps[i])]
            time = temp['time(s)'].iloc[0]
            sliprate = temp['slipv1'].iloc[0]
            # sliprate = V.max()
            
            line.set_data([time/self.t_yr], [sliprate])

                        
            timetext.set_text("Y{:0>5.0f} D{:0>3.0f} - {:0>2.0f}:{:0>2.0f}:{:0>4.2f}".format( time/(365*3600*24),
                                                  (time/3600/24)%(365),
                                                  (time/3600)%24,
                                                  (time/60)%60,
                                                  time%60)
                        )

            return surf, timetext, line
            # except Exception as e:
            #     print(e)
            #     pass
        
        anim = FuncAnimation(fig, update, frames=np.arange(1,self.N_steps-1,interval), blit=True)
        writer = animation.PillowWriter(fps=5)

        anim.save(os.path.join(self.path,"animation.gif"), 
                  writer=writer)
        plt.close()
        
        
        
    def animation3D_1(self, vmin = -8, vmax = -2, azim = 20, interval = 5):
        
        # --- Plot with Matplotlib ---
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=10, azim=azim, roll=0) 
        ax.set_box_aspect([2, 0.7, 1]) 

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        mesh = pv.read(os.path.join(self.out_folder, f'step{self.steps[0]}.vtu'))
    
        # Extract vertex coordinates
        points = mesh.points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
        tri_cells = mesh.extract_cells(np.where(mesh.celltypes == pv.CellType.TRIANGLE)[0])
        triangles = tri_cells.cells.reshape(-1, 4)[:, 1:4]  # skip the first '3' entry
        
        
        V = mesh.cell_data['Slipv[m/s]']
        
        surf = ax.plot_trisurf(
            x, y, z,
            triangles=triangles,
            cmap="magma",
            linewidth=0.1,
            edgecolor="none",
            alpha=1.0
        )
        
        
        surf.set_array(V)
        surf.set_norm(LogNorm(vmin=10**vmin, vmax=10**vmax))
        fig.colorbar(surf, ax=ax, shrink=0.3, label="V[m/s]", 
                     orientation = 'horizontal',
                     location='bottom',pad=-0.05, 
                     extend='both'
                     )
    
        
        def update(i):
            
            step = self.steps[i]
            print(step)
            mesh = pv.read(os.path.join(self.out_folder, f'step{step}.vtu'))
            V = mesh.cell_data['Slipv[m/s]']
    
            surf.set_array(V)
    
            return surf,
        
        anim = FuncAnimation(fig, update, frames=np.arange(1,self.N_steps,interval), blit=True)
        
        anim.save(os.path.join(self.path, "animation_1.mp4"), fps=20, dpi=150)
    
# sim_folder = sys.argv[1]
interval = int(sys.argv[2])

p = Ptool(sim_folder)
# p.plot_timeseries()
p.animation3D(azim = -80, interval = 5)

