#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:16:59 2025
This tool includes useful tools for plotting PyQuake3D outputs
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
    
    t_yr = 365*3600*24  # year to second converion
    
    Vdyn = 1e-3 ## Dynamic slip rate
    
    # header for outut file 
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
        self.out_folder = os.path.join(path, 'out_vtk') # output folder
        
        # get the output files
        self.vtk_files = [int(x.split('.vtu')[0][4:]) for x in os.listdir(self.out_folder) if x.endswith('.vtu') ]
        
        # Sort the files in the time order
        self.steps = np.sort(self.vtk_files)
        
        # State file is for maximum slip rate output
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
        '''
        This module plots time vs maximum slip rate for each the output 
        time steps

        Returns
        -------
        Saves the figure into the simulation directory.

        '''
        fig,ax = plt.subplots(1,1, figsize = (10,6), clear=True)
        
        vmax = self.read_statefile()

        ax.set_xlabel('time [yr]')
        ax.set_ylabel('log($V_{max}$) [m/s]')
        ax.semilogy(vmax['time(s)']/self.t_yr, 
                    np.sqrt(vmax['slipv1']**2 + vmax['slipv2']**2) 
                    )
        
        fig.savefig(os.path.join(self.path,'max_time_series.jpg'), dpi = 300, bbox_inches='tight')
        
    def animation2D(self, vmin = -9, vmax = 0, px='XZ', N_interval = 10):
        '''
        

        Parameters
        ----------
        vmin : float or integer, optional
            log10(minimum slip rate). The default is -9 => 10^(-9) m/s.
        vmax : float or integer, optional
            log10(maximum slip rate). The default is 0 => 10^0= 1 m/s.
        px : string, optional
            XZ: plot in (X,Z) domain, YZ plot in domain (Y,Z). 
            The default is 'XZ'.
        N_interval: integer, optional
            intervals for animation.

        Returns
        -------
        Animation saved to your simulation folder
        '''
        
        
        # --- Plot with Matplotlib ---
        # We have two subplots: 
        # Top : Slip rate plotted with the scatter on PX domain
        # Bottom : Maximim slip rate plot.
        fig,(ax,ax1) = plt.subplots(2,1, figsize = (8,6))
        

            
        # Read maximum slip rate file
        df = self.read_statefile()
        df['slipv'] = np.sqrt(df['slipv1']**2+df['slipv2']**2)
        # max_sliprate = np.sqrt(df['slipv1']**2 + df['slipv2']**2)
        
        
        # This is plot for the maximum slip rate
        ax1.set_xlabel('time [yr]')
        ax1.set_ylabel('V [m/s]')
        ax1.semilogy(df['time(s)']/self.t_yr, 
                    df['slipv'], 
                    lw = 1)
        
          
        
        # A red dot shows the maximum slip rate (bottom subplot), that is 
        # synchronized with the upper scatter plot, colored with slip rates.
        line, = ax1.semilogy(df['time(s)'].iloc[0]/self.t_yr, 
                    df['slipv'].iloc[0], color = 'r', marker = 'o')
        
        # This is the time information
        timetext = ax1.text(0.0,1.0, "Y{:0>5.0f} D{:0>3.0f}-{:0>2.0f}:{:0>2.0f}:{:0>2.0f}".format(0,
                                                  0,
                                                  0,
                                                  0,
                                                  0),
                           horizontalalignment='left',
                            verticalalignment='bottom',
                            transform = ax.transAxes)

        
        mesh = pv.read(os.path.join(self.out_folder, f'step{self.steps[0]}.vtu'))
        cells = mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
        triangles = cells[:, 1:]         # drop the "3"
        points = mesh.points[triangles]  # shape (n_cells, 3, 3)
        points = np.mean(points, axis = 1 )
        Z = points[:,2]
        
        V = mesh.cell_data['Slipv[m/s]']

        if px=='XZ':
            X = points[:,0]
            ax.set_xlabel('X[km]')
            ax.set_ylabel('Z[km]')
        else:
            X = points[:,1]
            ax.set_xlabel('Y[km]')
            ax.set_ylabel('Z[km]')     
            
        log_norm = LogNorm(vmin=10**vmin, vmax=10**vmax)
        
        sctr = ax.scatter(X*1e-3, Z*1e-3, c=V, cmap='magma', norm=log_norm, 
                          s = 5, edgecolor='none',
                          )
        
        cbar = fig.colorbar(sctr, ax=ax, label='V [m/s]', shrink = 0.5)

        def update(i):
            
            step = int(self.steps[i])
            print(step)
            
            mesh = pv.read(os.path.join(self.out_folder, f'step{step}.vtu'))
            V = mesh.cell_data['Slipv[m/s]']
            # sctr.set_offsets(np.c_[x_data[:frame+1], y_data[:frame+1]])
            sctr.set_array(V)

            temp = df[df.Iteration==int(self.steps[i])]
            time = temp['time(s)'].iloc[0]
            sliprate = temp['slipv'].iloc[0]
            
            line.set_data([time/self.t_yr], [sliprate])

                        
            timetext.set_text("Y{:0>5.0f} D{:0>3.0f} - {:0>2.0f}:{:0>2.0f}:{:0>4.2f}".format( time/(365*3600*24),
                                                  (time/3600/24)%(365),
                                                  (time/3600)%24,
                                                  (time/60)%60,
                                                  time%60)
                        )

            return sctr, timetext, line
        
        anim = FuncAnimation(fig, update, frames=np.arange(2,self.N_steps,N_interval), 
                             blit=True, )
        writer = animation.PillowWriter(fps=10)

        anim.save(os.path.join(self.path,"animation2D.gif"), writer = writer)        
        
        
    def animation3D(self, vmin = -10, vmax = 0, azim = 30, N_interval = 2):
        
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
        
        anim = FuncAnimation(fig, update, frames=np.arange(2,self.N_steps-1,N_interval), 
                             blit=True)
        writer = animation.PillowWriter(fps=20)

        anim.save(os.path.join(self.path,"animation.gif"), 
                  writer=writer)
        plt.close()
        
        
        
    def animation3D_1(self, vmin = -8, vmax = -2, azim = 20, N_interval = 5):
        '''
        This module plots without using subplots. if you want a subplot,
        use animation3D instead.

        Parameters
        ----------
        vmin : TYPE, optional
            DESCRIPTION. The default is -8.
        vmax : TYPE, optional
            DESCRIPTION. The default is -2.
        azim : TYPE, optional
            DESCRIPTION. The default is 20.
        interval : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        surf : TYPE
            DESCRIPTION.

        '''
        
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
        
        anim = FuncAnimation(fig, update, frames=np.arange(1,self.N_steps-1,N_interval), blit=True)
        
        anim.save(os.path.join(self.path, "animation_1.mp4"), fps=20, dpi=150)
        
        
    def extract_slip_info(self, V_dyn= 1e-3):
        '''
        This module reads PyQuake3D results recursively finds the slip events,
        then extract information about the slip event. 
        
        Vdyn : float, default:1e-3
                Dynamic slip rate. Quasi-dyanmic effect dominates the sloution.
        Returns
        -------
        None.

        '''
        
        

        df = self.read_statefile()
        df['slip_v'] = np.sqrt(df.slipv1**2+df.slipv2**2)
        
        # Vdyn= 5e-4
        
        df1 = df[df.slip_v>self.Vdyn]
        ind_temp = df1.index.diff(); 
        ind_temp2 = np.argwhere(ind_temp>1).flatten()
        Nevents = ind_temp2.size
        
        i_steps = self.steps
        
        event_string=f'{"Evnt":5}{"Nuc_X":10}{"Nuc_y":10}{"Nuc_z":10}{"X_min":10}{"X_max":10}{"Y_min":10}{"Y_max":10}{"Z_min":10}{"Z_max":10}{"slip_mean":10}{"slip_max":10}{"State_mean":16}{"State_min":16}{"Shear_mean":16}{"Shear_min":16}\n'
        
        ## Loop over events
        for i in range(Nevents-1):
            
            print(f'event {i+1}')
            # Finding indcies of the slip events form maximum slip rate file
            ind1 = int(ind_temp2[i])
            ind2 = int(ind_temp2[i+1]) - 1  
            
            # Find the iteration step number 
            iter1 = df1.iloc[ind1].Iteration
            iter2 = df1.iloc[ind2].Iteration
            
            iter_indices = ((i_steps>=iter1-10) & (i_steps<=iter2))
            
            N_iter = self.steps[iter_indices].size
            
            try:
                ## Loop during the event
                for ii in [0, N_iter-1]:    
                    step = self.steps[iter_indices][ii]
                    
                    # read the output file depending on the iteration step
                    mesh = pv.read(os.path.join(self.out_folder, f'step{step}.vtu'))
                    
                    # Get data
                    cells = mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
                    triangles = cells[:, 1:]         # drop the "3"
                    points = mesh.points[triangles]  # shape (n_cells, 3, 3)
                    points = np.mean(points, axis = 1 )
                    V = mesh.cell_data['Slipv[m/s]']
                
                    # Find the index of slip rate exceeds dynamic slip rate
                    ind_Vdyn = (V > self.Vdyn)
                    X_min = points[ind_Vdyn,0].min() 
                    X_max = points[ind_Vdyn,0].max() 
                    
                    Y_min = points[ind_Vdyn,1].min() 
                    Y_max = points[ind_Vdyn,1].max() 
                    
                    Z_min = points[ind_Vdyn,2].min() 
                    Z_max = points[ind_Vdyn,2].max() 
                    
                    if ii == 0:
                        # Nucleation Point
                        Nuc = ((X_min+X_max)*0.5, (Y_min+Y_max)*0.5, (Z_min+Z_max)*0.5)
                        
                        # beginning of slip
                        slip_ini = mesh.cell_data['slip[m]']
                        shear_ini = mesh.cell_data['Shear_[MPa]']
                        state_ini = mesh.cell_data['state']
            
                    elif ii == N_iter - 1 :
                        # beginning of slip
                        slip_end = mesh.cell_data['slip[m]']
                        shear_end = mesh.cell_data['Shear_[MPa]']
                        state_end = mesh.cell_data['state']
            
                #VW index
                # a_min_b = mesh.cell_data['a-b']
                # ind_vw = a_min_b<0
                    
                slip_max = (slip_end - slip_ini).max() 
                slip_mean = (slip_end - slip_ini).mean()
                    
                state_min = (state_end - state_ini).min() 
                state_mean = (state_end - state_ini).mean()
                
                shear_min = (shear_end - shear_ini).min() 
                shear_mean = (shear_end - shear_ini).mean()
                    
                event_string += f'{i:5.0f}{Nuc[0]:10.1f}{Nuc[1]:10.1f}{Nuc[2]:10.1f}{X_min:10.1f}{X_max:10.1f}{Y_min:10.1f}{Y_max:10.1f}{Z_min:10.1f}{Z_max:10.1f}{slip_mean:10.3f}{slip_max:10.3f}{state_mean:16.6E}{state_min:16.6E}{shear_mean:16.6E}{shear_min:16.6E}\n'
    
            except Exception as e:
                print(e)
                pass
            
            
        with open(os.path.join(self.path, "events.txt"), "w") as file:
            file.write(event_string)

