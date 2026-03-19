#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:16:59 2025


PyQuake3D Visualization Utilities

This module provides tools for visualizing and analyzing
PyQuake3D simulation outputs, including:

- Time series plots of maximum slip rate
- 2D and 3D animations of fault slip
- Automatic earthquake event detection
- Slip statistics and seismic moment calculations



@author: eyup
"""

import pyvista as pv
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import traceback
from scipy.integrate import simpson
from scipy.interpolate import LinearNDInterpolator


class Ptool:
    """
    Visualization and analysis toolkit for PyQuake3D outputs.
    
    Parameters
    ----------
    path : str
        Path to the simulation directory containing output files.
    
    Notes
    -----
    The tool expects the following directory structure:
    
        simulation/
        ├── state.txt
        ├── out_vtu/
        │   ├── step0.vtu
        │   ├── step1.vtu
        │   └── ...
    """
    
    # ------------------------------------------------------------------
    # Physical constants
    # ------------------------------------------------------------------
    G = 32038120320                 # Shear Modulus
    rho = 2670                      # Density of the rocth
    c_s = 0.5*(G/rho)**(1/2)        # Shear Wave Velocity
    V_dyn = 1e-2    ## Dynamic slip rate, when elastodynamic effects dominate
    t_yr = 365*3600*24  # year to second converion

    # ------------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------------
    
    var = 'V'       # Variable to be plotted. 
    t_min = 0       # Minimum time to be plotted 
    t_max = 1e100       # Maximum time to be plotted 
    V_min = -8       # Minimum slip rate to be plotted (LOG10 SCALE)
    V_max = 0        # Maximum slip rate to be plotted (LOG10 SCALE)
    Omega_min = -1       # Minimum Omega to be plotted (LOG10 SCALE)
    Omega_max = 1        # Maximum Omega to be plotted (LOG10 SCALE)
    theta_min = 0      # Minimum theta to be plotted (LOG10 SCALE)
    theta_max = 1        # Maximum theta to be plotted (LOG10 SCALE)
    azimuth = -80        # Azimuth angle for 3D plot
    elevation = 15       # Elevation angle for 3D plot
    interval = 10        # Interval of reading outputs for animations
    depth = 10e3        # Depth for event plot 
    event_no = 3        # Event number to be plotted
    next_event = 0      # This number is used for plotting how manu number of events


    # ------------------------------------------------------------------
    # Output fields
    # ------------------------------------------------------------------

    field = [
        "Normal_[MPa]",
        "Pore_pressure[MPa]",
        "Shear_[MPa]",
        "Shear_1[MPa]",
        "Shear_2[MPa]",
        "rake[Degree]",
        "state",
        "Slipv[m/s]",
        "Slipv1[m/s]",
        "Slipv2[m/s]",
        "a",
        "b",
        "a-b",
        "dc",
        "fric",
        "slip",
        "slip1",
        "slip2",
        "slip_plate",
    ]
    

    def __init__(self, path):
    
        """Initialize the visualization tool."""

        self.path = path

        out_vtu = os.path.join(path, "out_vtu")
        out_vtk = os.path.join(path, "out_vtk")

        if os.path.isdir(out_vtu):
            self.out_folder = out_vtu
            self.extension = ".vtu"
        elif os.path.isdir(out_vtk):
            self.out_folder = out_vtk
            self.extension = ".vtu"
        else:
            raise FileNotFoundError(
                f"No output folder found. Checked: {out_vtu} and {out_vtk}"
            )

        files = [
            int(f.split(self.extension)[0][4:])
            for f in os.listdir(self.out_folder)
            if f.endswith(self.extension)
        ]

        self.steps = np.sort(files)
        self.N_steps = len(self.steps)

        self.state_file = os.path.join(path, "state.txt")
        
        
        '''
        This function reads the initial variables and frictional parameters
        Returns
        -------
        Saves to the class object.

        '''
        init_mesh = pv.read(
            os.path.join(self.path, 'Init.vtu')
            )
        self.Init_mesh = init_mesh
        
        # Get data
        cells = init_mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
        triangles = cells[:, 1:]         # drop the "3"
        points = init_mesh.points[triangles]  # shape (n_cells, 3, 3)
        points = np.mean(points, axis = 1 )
        
        ## Center of the trinages in X, Y, Z 
        self.x, self.y, self.z = points[:,0],points[:,1], points[:,2]  
        
        self.x_min = self.x.min() 
        self.x_max = self.x.max() 
        self.x_mean= self.x.mean() 

        self.y_min = self.y.min() 
        self.y_max = self.y.max() 
        self.y_mean= self.y.mean() 
        
        self.z_min = self.z.min() 
        self.z_max = self.z.max() 
        self.z_mean= self.z.mean()         
        
        
    # ----------------------------------------------------------------------- #
    #                           HELPER FUNCTIONS                              #
    # ----------------------------------------------------------------------- #

    def read_mesh(self, step):
        """Load mesh for a given simulation step."""
        return pv.read(os.path.join(self.out_folder, f"step{step}.vtu"))

            
    
    def read_statefile(self):
        try:
            vmax = pd.read_csv(
            self.state_file,
            sep = '\\s+', skiprows=15,low_memory=True,
            names=['Iteration', 'dt', 'slipv1', 'slipv2', 'time(s)', 'time(h)']
           	     )
        except:
            vmax = pd.read_csv(
            self.state_file,
            sep = '\\s+', skiprows=18,low_memory=True,
            names=['Iteration', 'dt', 'slipv1', 'slipv2', 'time(s)', 'time(h)'],
            skipfooter=0, engine='python' )
	
        vmax = vmax.dropna().astype(float)
        return vmax
    
    def seismic_moment(self, time, MO_dot):
        '''
        

        Parameters
        ----------
        time : TYPE
            DESCRIPTION.
        M0_dot : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        # Compute seismic moment and 
        M0 = np.abs(simpson(MO_dot, x=time))
        Mw = 2/3 * (np.log10(M0) - 9.1)
        M0_dot_mean = 10**np.mean(np.log10(MO_dot))
        
        return (M0, Mw, M0_dot_mean)
    
    def plot_timeseries(self):
        '''
        This module plots time vs maximum slip rate for each the output 
        time steps

        Returns
        -------
        Saves the figure into the simulation directory.

        '''

        
        fig,ax = plt.subplots(1,1, figsize = (10,6), clear=True)
        
        ax.set_yscale('log')
    
        vmax = self.read_statefile()
        vmax['Slipv[m/s]'] = np.sqrt(vmax['slipv1']**2 + vmax['slipv2']**2)

        ax.set_xlabel('time [yr]')
 
        if self.var == 'V':
            ax.set_ylabel('log($V_{max}$) [m/s]')
            # V = np.sqrt(vmax['slipv1']**2 + vmax['slipv2']**2)
            V = vmax['Slipv[m/s]']
            ax.plot(vmax['time(s)']/self.t_yr, 
                        V)
            ax.set_ylabel('log($V_{max}$) [m/s]')

        if self.var == 'theta':
            ax.set_ylabel('$\\Omega [-]$')
            ax.plot(vmax['time(s)']/self.t_yr, 
                        vmax['theta'])
            ax.set_ylabel('$\\theta$ [s]')
            
        if self.var == 'fric':
            ax.set_ylabel('Slip [m]$')
            ax.set_yscale('linear')
            ax.plot(vmax['time(s)']/self.t_yr, 
                        vmax['fric'])
            
        if self.var == 'shear':
            ax.set_ylabel('Slip [m]$')
            ax.set_yscale('linear')
            ax.plot(vmax['time(s)']/self.t_yr, 
                        vmax['Shear_[MPa]'])


        fig.savefig(os.path.join(self.path,f'max_{self.var}.jpg'), 
                    dpi = 300, bbox_inches='tight')
        


    def plot_timeseries2(self):
        '''
        This module plots time vs maximum slip rate for each the output 
        time steps

        Returns
        -------
        Saves the figure into the simulation directory.

        '''

        # read max file
        vmax = self.read_statefile()
        
        # read Event file 
        event = pd.read_csv(
            os.path.join(self.path, 'events.txt'), sep = '\\s+'
            )
        
        event_no2 = self.event_no+self.next_event
        s_event1 = event[event['Evnt'] == self.event_no]
        s_event2 = event[event['Evnt'] == event_no2]

        
        I_start = s_event1['I_start'].values[0]
        
        I_finish = s_event2['I_finish'].values[0]
        
        selected_steps = self.steps[(self.steps>=I_start) & 
                                    (self.steps<I_finish)]
        
        V_max = np.empty(selected_steps.size)
        V_mean = np.empty(selected_steps.size)

        Psi = np.empty(selected_steps.size)
        Fric = np.empty(selected_steps.size)

        P_max = np.empty(selected_steps.size)
        P_mean = np.empty(selected_steps.size)

        Por = np.empty(selected_steps.size)

        S = np.empty(selected_steps.size)
        T = np.empty(selected_steps.size)
        Time = np.empty(selected_steps.size)

        i = 0
        
        for step in selected_steps:
            
            Time[i] = vmax[vmax.Iteration==step]['time(s)'].values[0]

            vtu_file = f'{self.out_folder}/step{step}{self.extension}'
            
            mesh_v = pv.read(vtu_file)
            
            V_max[i] = mesh_v.cell_data["Slipv[m/s]"].max()
            V_mean[i] = mesh_v.cell_data["Slipv[m/s]"].mean()

            Psi[i] = mesh_v.cell_data["state"].mean()
            Fric[i] = mesh_v.cell_data["fric"].mean()

            P_max[i] = mesh_v.cell_data['Pore_pressure[MPa]'].max()
            P_mean[i] = mesh_v.cell_data['Pore_pressure[MPa]'].mean()

            S[i] = mesh_v.cell_data['Normal_[MPa]'].mean()
            Por[i] = mesh_v.cell_data['Porosity[Degree]'].mean()
            T[i] = mesh_v.cell_data['Temperature[Degree]'].mean()

            i+=1
                
        Time = Time/365/3600/24
        fig,(ax,ax1,ax2) = plt.subplots(3,1, figsize = (10,12), sharex =True, clear=True)
        
        
        ax.set_ylabel('V')
        ax1.set_ylabel('$friction$')
        ax2.set_ylabel('[MPa]')
        ax2.set_xlabel('Time [year]')
        
        ax.semilogy(Time, V_max, label='V_max')
        ax.semilogy(Time, V_mean, label='V_mean')
        
        ax1.plot(Time, Fric, label='friction')
        ax1.plot(Time, Psi, label='Restrengthening')
        
        ax2.semilogy(Time, S-P_max, label='$\\sigma_n - P_{max}$ [MPa]')
        ax2.semilogy(Time, S-P_mean, label= '$\\sigma_n - P_{mean}$ [MPa]')
        
        ax4 = ax2.twinx()
        ax4.plot(Time, T, color = 'k', lw = 1, label='Temperature [Degree]')
        ax4.set_ylabel('Temperature')
        
        ax4.set_ylim(top=50)

        ax.legend()
        ax1.legend()
        ax2.legend()
        
        fig.savefig(os.path.join(self.path, f'ts2_{self.event_no}-{event_no2}.jpg'), 
                    dpi = 300, bbox_inches='tight')
        



        
        
    def animation2D(self):
        '''
        

        Parameters
        ----------
        Returns
        -------
        Animation saved to your simulation folder
        '''
        px='XZ'
        
        # --- Plot with Matplotlib ---
        # We have two subplots: 
        # Top : Slip rate plotted with the scatter on PX domain
        # Bottom : Maximim slip rate plot.
        fig,(ax,ax1) = plt.subplots(2,1, figsize = (8,6))
        

            
        # Read maximum slip rate file
        df = self.read_statefile()
        df['slipv'] = np.sqrt(df['slipv1']**2+df['slipv2']**2)

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
        timetext = ax1.text(0.0,1.0, 
                "Y{:0>5.0f} D{:0>3.0f}-{:0>2.0f}:{:0>2.0f}:{:0>2.0f}".format(0,
                                                  0,
                                                  0,
                                                  0,
                                                  0),
                           horizontalalignment='left',
                            verticalalignment='bottom',
                            transform = ax.transAxes)

        
        mesh = self.read_mesh(step=0)
        
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
            
        log_norm = LogNorm(vmin=10**self.V_min, vmax=10**self.V_max)
        
        sctr = ax.scatter(X*1e-3, Z*1e-3, c=V, cmap='magma', norm=log_norm, 
                          s = 5, edgecolor='none',
                          )
        
        cbar = fig.colorbar(sctr, ax=ax, label='V [m/s]', shrink = 0.5)

        def update(i):
            
            step = int(self.steps[i])
            print(step)
            
            mesh = self.read_mesh(step = step)
            
            V = mesh.cell_data['Slipv[m/s]']

            sctr.set_array(V)

            temp = df[df.Iteration==int(self.steps[i])]
            time = temp['time(s)'].iloc[0]
            sliprate = temp['slipv'].iloc[0]
            
            line.set_data([time/self.t_yr], [sliprate])

                        
            timetext.set_text(
                "Y{:0>5.0f} D{:0>3.0f} - {:0>2.0f}:{:0>2.0f}:{:0>4.2f}".format( time/(365*3600*24),
                                                  (time/3600/24)%(365),
                                                  (time/3600)%24,
                                                  (time/60)%60,
                                                  time%60)
                        )

            return sctr, timetext, line
        
        anim = FuncAnimation(fig, update, frames=np.arange(2,self.N_steps,self.interval), 
                             blit=True, )
        writer = animation.PillowWriter(fps=10)

        anim.save(os.path.join(self.path,"animation2D.gif"), writer = writer)        
        
        
    def animation3D(self):
        

        # --- Plot with Matplotlib ---
        fig = plt.figure(figsize=(10,8))
        
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.05, 0.35, 0.9, 0.7])
        ax.view_init(elev=self.elevation, azim=self.azimuth) 
        ax.set_box_aspect([1.6, 0.8, 0.6]) 
        ax1 = fig.add_subplot(611)
        ax1.set_position([0.1, 0.05, 0.85, 0.2])
        
        df = self.read_statefile()
        df = df.loc[(df['time(s)']/self.t_yr<self.t_max)]
        df = df.loc[(df['time(s)']/self.t_yr>self.t_min)]
	
        step_min = df.Iteration.min()
        step_max = df.Iteration.max() 
        
        # Filter steps 
        
        steps_filtered = self.steps[(self.steps>step_min) & (self.steps<step_max)]
        N_steps = len(steps_filtered)
        
        
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
        
        mesh = self.read_mesh(step = 0)


        # Extract vertex coordinates
        points = mesh.points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        tri_cells = mesh.extract_cells(np.where(mesh.celltypes == pv.CellType.TRIANGLE)[0])
        triangles = tri_cells.cells.reshape(-1, 4)[:, 1:4]  # skip the first '3' entry
        
        
        
        surf = ax.plot_trisurf(
            x*1E-3, y*1E-3, z*1E-3,
            triangles=triangles,
            # linewidth=0.05,
            edgecolor="none",
            alpha=1.0
        )
        
        
        
        if self.var =='V':
            
            label = 'V [m/s]'
            cmap = 'magma'
            dummy = 'Slipv[m/s]'
            data = mesh.cell_data[f'{dummy}']
            surf.set_array(data)

            norm = LogNorm(vmin=10**self.V_min, 
                           vmax=10**self.V_max)

            surf.set_norm(norm)
        elif self.var =='Omega':
            label = '$\\Omega$ [-]'
            cmap = 'seismic'
            data = mesh.cell_data['Slipv[m/s]'] * mesh.cell_data['state'] / mesh.cell_data['dc']
            surf.set_array(data)

            norm = LogNorm(vmin=10**self.Omega_min, 
                           vmax=10**self.Omega_max)
            surf.set_norm(norm)
        elif self.var =='state':
            label = '$\\phi$ [-]'
            dummy = 'state'
            data = mesh.cell_data['state']
            surf.set_array(data)
            # norm = LogNorm(vmin=1e-5, 
            #                vmax=1)
            # boundaries = np.linspace(self.thet
            # norm = LogNorm(vmin=self.theta_min, 
            #                vmax=self.theta_max)
            boundaries = np.linspace(0.1,1, 
                                     num=100,endpoint=True)
            norm = BoundaryNorm(boundaries, ncolors=256, extend= 'both')
            cmap = 'plasma_r'
            surf.set_cmap(cmap)
            surf.set_norm(norm)   

        fig.colorbar(surf, ax=ax, shrink=0.3, 
                     label=f"{label}", 
                     orientation = 'vertical',
                     location='right', pad = 0.1,
                     extend='both',
                     norm=norm
                     )
        
        
        def update(i):

            step = steps_filtered[i]
            
            print('step: {:<10.0f} year: {:<10.4f}'.format(step, df.iloc[i]['time(s)']/self.t_yr))
            
            
            mesh = self.read_mesh(step = step)
            
            if self.var !='Omega':
                data = mesh.cell_data[f'{dummy}']
            else:
                data = mesh.cell_data['Slipv[m/s]'] * mesh.cell_data['state'] / mesh.cell_data['dc']
            
            temp = df[df.Iteration==step]
            time = temp['time(s)'].iloc[0]
            sliprate = temp['slipv1'].iloc[0]            
            line.set_data([time/self.t_yr], [sliprate])

                        
            timetext.set_text("Y{:0>5.0f} D{:0>3.0f} - {:0>2.0f}:{:0>2.0f}:{:0>4.2f}".format( time/(365*3600*24),
                                                  (time/3600/24)%(365),
                                                  (time/3600)%24,
                                                  (time/60)%60,
                                                  time%60)
                        )
            
            surf.set_array(data)
            # surf.set_norm(norm)
            return surf, timetext, line

        
        anim = FuncAnimation(fig, update, 
                             frames=np.arange(0,N_steps,self.interval), 
                             blit=True)
        
        writer = animation.PillowWriter(fps=10)

        anim.save(os.path.join(self.path, f"animation_{self.var}.gif"), 
                  writer = writer)
        plt.close()
        
        
    def extract_slip_info(self):
        '''
        This module reads PyQuake3D results recursively finds the slip events,
        then extract information about the slip event. 
        
        Returns
        -------
        None.

        '''        
        dt_crit = 1e3 # critical time to distinguish slip events

        df = self.read_statefile()
        df['slip_v'] = np.sqrt(df.slipv1**2+df.slipv2**2)
                
        df1 = df[df.slip_v>self.V_dyn]
        
        dtime = np.diff(df1['time(s)'].to_numpy(), prepend=0)
        dtime = np.append(dtime, dt_crit)

        ind_temp = np.argwhere(dtime>=dt_crit).flatten()
        Nevents = ind_temp.size
        
        i_steps = self.steps
        
        event_string=f'{"Evnt":10}{"I_start":10}{"I_finish":10}{"Year":10}{"Day":10}{"Hour":10}{"Min":10}{"Duration":10}{"Nuc_X":10}{"Nuc_y":10}{"Nuc_z":10}{"X_min":10}{"X_max":10}{"Y_min":10}{"Y_max":10}{"Z_min":10}{"Z_max":10}{"slip_mean":10}{"slip_max":10}{"State_drop":16}{"Stress_drop":16}{"M0":16}{"M0_dot_mean":16}{"Mw":16}\n'
        
        ## Loop over events
        for i in range(Nevents-1):
            
            print(f'event {i+1}')
            # Finding indcies of the slip events form maximum slip rate file
            ind1 = int(ind_temp[i])
            ind2 = int(ind_temp[i+1]) - 1  
            
            # Find the iteration step number 
            iter1 = df1.iloc[ind1].Iteration
            iter2 = df1.iloc[ind2].Iteration
            
            # earthquake_ind = df[(df.Iteration>=iter1) & (df.Iteration<=iter1)]['slip_v'].argmax()
            
            # Get the indices of the event
            iter_indices = ((i_steps>=iter1) & (i_steps<=iter2))
            step_min = self.steps[iter_indices].min() # Start index of the event
            step_max = self.steps[iter_indices].max() # Finish index of the event
            N_iter = self.steps[iter_indices].size
            
            X_min = []
            X_max = []
            Y_min = []
            Y_max = []
            Z_min = []
            Z_max = []
            
            
            MO_dot = np.empty(N_iter)
            time = np.empty(N_iter)
            # V_event = np.empty(N_iter)

            try:
                ## Loop during the event
                for ii in range(N_iter):  
                    step = self.steps[iter_indices][ii]

                    # Get time
                    time[ii] = df[df.Iteration==step]['time(s)'].values
                    # V_event[ii] == df[df.Iteration==step]['slip_v'].values
                    # read the output file depending on the iteration step
                    
                    mesh = self.read_mesh(step = step)

                    # Get data
                    cells = mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
                    triangles = cells[:, 1:]         # drop the "3"
                    points = mesh.points[triangles]  # shape (n_cells, 3, 3)
                    points = np.mean(points, axis = 1 )
                    
                    mesh_with_areas = mesh.compute_cell_sizes(area=True, volume=False)
                    
                    V = mesh.cell_data['Slipv[m/s]']
                    A = mesh_with_areas.cell_data['Area']
                    
                    # Seismic moment release rate
                    MO_dot[ii] = np.abs(np.sum(A*V*self.G))
                                    
                    # Find the index of slip rate exceeds dynamic slip rate
                    ind_Vdyn = (V > self.V_dyn)
                                        
                    
                    X_min1 = points[ind_Vdyn,0].min()
                    X_min.append(X_min1)
                    X_max1 = points[ind_Vdyn,0].max()
                    X_max.append(X_max1)
                    Y_min1 = points[ind_Vdyn,1].min()
                    Y_min.append(Y_min1)
                    Y_max1 = points[ind_Vdyn,1].max()
                    Y_max.append(Y_max1)
                    Z_min1 = points[ind_Vdyn,2].min()
                    Z_min.append(Z_min1)
                    Z_max1 = points[ind_Vdyn,2].max()
                    Z_max.append(Z_max1)
                    
                    if ii == 0:
                        # Nucleation Point
                        Nuc = ((X_min1+X_max1)*0.5, (Y_min1+Y_max1)*0.5, (Z_min1+Z_max1)*0.5)
                        
                        # beginning of slip
                        # Find the index of maximum state. At the end of the 
                        # rupture we will compare the stress drop

                        # max_state_ind = mesh.cell_data['state'].argmax()
                        slip_ini = mesh.cell_data['slip[m]']
                        shear_ini = mesh.cell_data['Shear_[MPa]']
                        state_ini = mesh.cell_data['state']
            
                    elif ii == N_iter - 1 :
                        # beginning of slip
                        slip_end = mesh.cell_data['slip[m]']
                        shear_end = mesh.cell_data['Shear_[MPa]']
                        state_end = mesh.cell_data['state']
            
                    
                # Seismic moment info
                (M0, Mw, M0_dot_mean) = self.seismic_moment(time, MO_dot)
                
                
                slip_max = (slip_end - slip_ini).max() 
                slip_mean = (slip_end - slip_ini).mean()                    
                state_drop = (state_end - state_ini).mean()
                stress_drop = (shear_end - shear_ini).mean() 
                
                X_min = np.min(X_min)
                X_max = np.max(X_max)
                Y_min = np.min(Y_min)
                Y_max = np.max(Y_max)
                Z_min = np.min(Z_min)
                Z_max = np.max(Z_max)
                
                
                # ttime = time[vv.argmax()]
                ttime0 = time[0]
                ttime1 = time[-1]
                t_year = ttime0 // self.t_yr 
                t_day  = (ttime0 / 3600 / 24 ) % 365
                t_hour = (ttime0 / 3600 ) % 24
                t_min  = (ttime0 / 60 ) % 60
                t_dur = ttime1 - ttime0
                
                event_string += f'{i:5.0f}{step_min:10.0f}{step_max:10.0f}{t_year:10.0f}{t_day:10.0f}{t_hour:10.0f}{t_min:10.2f}{t_dur:10.2f}{Nuc[0]:10.1f}{Nuc[1]:10.1f}{Nuc[2]:10.1f}{X_min:10.1f}{X_max:10.1f}{Y_min:10.1f}{Y_max:10.1f}{Z_min:10.1f}{Z_max:10.1f}{slip_mean:10.3f}{slip_max:10.3f}{state_drop:16.6E}{stress_drop:16.6E}{M0:16.6E}{M0_dot_mean:16.6E}{Mw:16.3f}\n'

                
            except Exception as e:
                print(e)
                print(traceback.print_exc()) # Prints the full traceback to stderr

                pass
            
            
        with open(os.path.join(self.path, "events.txt"), "w") as file:
            file.write(event_string)
            
            
    def phase_plot(self):
        '''
        This is a phase plot Velocity versus Psi
        Psi=f_0 + b * ln(theta V_0/dc)

        Returns
        -------
        Plot.

        '''
        
        fig,ax = plt.subplots(1,1)
        ax.set_xlabel('V [m/s]')
        ax.set_ylabel('$\\psi$')
        
        Vmean_data = np.empty(self.N_steps)
        statemean_data = np.empty(self.N_steps)

        mesh_init = pv.read(os.path.join(self.path, 'Init.vtu'))
        a = mesh_init.cell_data['a'] 
        b = mesh_init.cell_data['b'] 
        dc = mesh_init.cell_data['dc'] 
        i = 0
        for step in self.steps:
            # print(step)
            mesh = self.read_mesh(step)
            Vmean_data[i] = mesh.cell_data['Slipv[m/s]'].mean()
            statemean_data[i] = mesh.cell_data['state'].mean()    
            i+=1
            # ax.scatter(np.log10(V), state) 
            


        ax.semilogx(Vmean_data, statemean_data, color = 'k', lw = 0.8) 
        
        fig.savefig(os.path.join(self.path,'phase_plot.jpg'), 
                dpi = 300, bbox_inches='tight')


    def event_plot(self):
        
        # read max file
        vmax = self.read_statefile()
        
        # read Event file 
        event = pd.read_csv(
            os.path.join(self.path, 'events.txt'), sep = '\\s+'
            )
        
        s_event = event[event['Evnt'] == self.event_no]
        
        I_start = s_event['I_start'].values[0]
        
        I_finish = s_event['I_finish'].values[0]
        
        selected_steps = self.steps[(self.steps>=I_start) & 
                                    (self.steps<I_finish)]
        
        pp = np.stack([self.x,self.z]).T
        
        x_fine = np.linspace(self.x_min+10,self.x_max-10,1000, 
                             endpoint=True)
        
        # Depth to be plotted!
        z_depth = -self.depth
        
        V_m = np.empty((selected_steps.size, x_fine.size))
        time_m = np.empty((selected_steps.size)).flatten()
        
        i = 0
        
        for step in selected_steps:
            
            vtu_file = f'{self.out_folder}/step{step}{self.extension}'
            
            time = vmax[vmax.Iteration==step]['time(s)'].values[0]
            print(time, step)
            
            mesh_v = pv.read(vtu_file)
            slip_rate = mesh_v.cell_data["Slipv[m/s]"]
            
            p_fine = np.stack([x_fine, np.ones(x_fine.size)*z_depth]).T 
            
            interp = LinearNDInterpolator(pp, slip_rate)
            
            V_m[i,:] = interp(p_fine)
                
            time_m[i] = time   
            
            i+=1
        
        V_m[:,0] = V_m[:,1] 
        
        
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Position [km]')
        ax.set_ylabel('Time [s]')
        
        t1 = np.arange(0,10) 
        x1 = t1*self.c_s
        
        levs = np.logspace(-8, 1, 50)
        
        
        # ax.set_ylim(bottom = 60)
        cs = ax.contourf(x_fine*1e-3, time_m, V_m, levels=levs, 
                         norm=LogNorm(), cmap='Reds') 
        ax.plot( (self.x_mean + x1) * 1e-3, t1 + time_m.min()+5,ls = '--', color = 'k')
        ax.plot( (self.x_mean - x1) * 1e-3, t1 + time_m.min()+5,ls = '--', color = 'k')
        
        cbar = fig.colorbar(cs, format='%.0e', 
                            shrink = 0.5, 
                            label = 'Slip rate [m/s]') 


        fig.savefig(os.path.join(self.path,f'event_{self.event_no}_depth{self.depth*1e-3:.0f}.jpg'), 
                    dpi = 200, bbox_inches='tight' )