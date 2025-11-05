#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 10:02:47 2025
This script is an example for mesh generation using gmsh
@author: eyup
"""
import os
import gmsh
import math
import sys
from scipy.interpolate import interp1d
import numpy as np
import sys
import matplotlib.pyplot as plt



# %%
"""
Necessary functions
"""

def arcsinh(x):
    return math.log(x + math.sqrt(x**2 + 1))


def fault_corners(x0, y0, z0, L, W, strike, dip):
    """
    Compute 4 corner coordinates of a dipping rectangular fault
    starting from down-dip left corner (x0,y0,z0)
    """
    # # =====================
    # # Helper functions
    # # =====================
    def deg2rad(a):
        return a * math.pi / 180.0
    
    strike_rad = deg2rad(strike)
    dip_rad = deg2rad(dip)

    # Strike direction (horizontal unit vector)
    sx = math.sin(strike_rad)
    sy = math.cos(strike_rad)

    # Dip direction (unit vector in 3D)
    dx = math.cos(dip_rad) * math.cos(strike_rad)
    dy = -math.cos(dip_rad) * math.sin(strike_rad)
    dz = -math.sin(dip_rad)  # dipping downwards

    # p1 = given point (down-dip left corner)
    p1 = (x0, y0, z0)
    # Along strike from p1 (bottom edge)
    # p2 = (p1[0] + L * sx, p1[1] + L * sy, p1[2])
    p2 = (p1[0] + L * sx, 0, p1[2])

    # Up-dip edge from p1
    # p4 = (p1[0] - W * dx, p1[1] - W * dy, p1[2] - W * dz)
    p4 = (round(p1[0] - W * dx,2), 0, p1[2] - W * dz)

    # Up-dip edge from p2
    # p3 = (p2[0] - W * dx, p2[1] - W * dy, p2[2] - W * dz)
    p3 = (p2[0] - W * dx, 0, p2[2] - W * dz)

    return [p1, p2, p3, p4]



def ouputVTK(fname):
    Nnode=nodelst.shape[0]
    Nele=elelst.shape[0]
    f=open(fname,'w')
    f.write('# vtk DataFile Version 3.0\n')
    f.write('test\n')
    f.write('ASCII\n')
    f.write('DATASET  UNSTRUCTURED_GRID\n')
    f.write('POINTS '+str(Nnode)+' float\n')
    for i in range(Nnode):
        f.write('%f %f %f\n'%(nodelst[i][0],nodelst[i][1],nodelst[i][2]))
    f.write('CELLS '+str(Nele)+' '+str(Nele*4)+'\n')
    for i in range(Nele):
        f.write('3 %d %d %d\n'%(elelst[i][0]-1,elelst[i][1]-1,elelst[i][2]-1))
    f.write('CELL_TYPES '+str(Nele)+'\n')
    for i in range(Nele):
        f.write('5 ')
    f.write('\n')
    

    f.write('CELL_DATA %d ' %(Nele))
    f.write('SCALARS Normal_[MPa] float\nLOOKUP_TABLE default\n')
    for i in range(len(Tno)):
        f.write('%f '%(Tno[i]))
    f.write('\n')

    f.write('SCALARS depele[km] float\nLOOKUP_TABLE default\n')
    for i in range(len(depele)):
        f.write('%f '%(depele[i]))
    f.write('\n')

    f.write('SCALARS a float\nLOOKUP_TABLE default\n')
    for i in range(len(alst)):
        f.write('%f '%(alst[i]))
    f.write('\n')

    f.write('SCALARS b float\nLOOKUP_TABLE default\n')
    for i in range(len(blst)):
        f.write('%f '%(blst[i]))
    f.write('\n')

    f.write('SCALARS a-b float\nLOOKUP_TABLE default\n')
    for i in range(len(blst)):
        f.write('%f '%(alst[i]-blst[i]))
    f.write('\n')

    f.write('SCALARS rake float\nLOOKUP_TABLE default\n')
    for i in range(len(rake)):
        f.write('%f '%(rake[i]))
    f.write('\n')
    
    f.write('SCALARS Dc float\nLOOKUP_TABLE default\n')
    for i in range(len(Dc)):
        f.write('%f '%(Dc[i]))
    f.write('\n')
    
    f.write('SCALARS Shear_[MPa] float\nLOOKUP_TABLE default\n')
    for i in range(len(Tt)):
        f.write('%f '%(Tt[i]))
    f.write('\n')

    f.write('SCALARS Shear_1[MPa] float\nLOOKUP_TABLE default\n')
    for i in range(len(Tt)):
        f.write('%f '%(Tt1o[i]))
    f.write('\n')

    f.write('SCALARS Shear_2[MPa] float\nLOOKUP_TABLE default\n')
    for i in range(len(Tt)):
        f.write('%f '%(Tt2o[i]))
    f.write('\n')
    
    
def read_mshV2(fname):  #read msh file
    f=open(fname,'r')
    start_node=0
    start_ele=0
    K1=0
    node=[]
    ele=[]
    for line in f:
        line=line[:-1]
        
        if(start_node==1):
            K1=K1+1
            sliptdata=line.split()
            #print(sliptdata)
            if(K1>1):
                try:
                         
                    nodeline=[]
                    tem=float(line.split()[1])
                    nodeline.append(tem)
                    tem=float(line.split()[2])
                    nodeline.append(tem)
                    tem=float(line.split()[3])
                    nodeline.append(tem)
                    node.append(nodeline)
                    #print(nodeline)
                except:
                    start_node=0
                    K1=0
        if(start_ele==1):
            K1=K1+1
            sliptdata=line.split()
            if(K1>1):
                try:
                    if(len(line.split())>=8): 
                        eleline=[]
                        tem=int(sliptdata[-3])
                        eleline.append(tem)
                        tem=int(sliptdata[-2])
                        eleline.append(tem)
                        tem=int(sliptdata[-1])
                        eleline.append(tem)
                        ele.append(eleline)
                        #print(eleline)
                except:
                    start_ele=0

        if(line=='$Nodes'):
            start_node=1
            
        if(line=='$Elements'):
            start_ele=1
            #print(start_ele)
        
        #boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(ele)
    
    return np.array(node),np.array(ele)
    

# fname='1.msh'
# nodelst,elelst=read_mshV2(fname)
# print(nodelst.shape,elelst.shape)

def get_eleVec(nodelst,elelst,jud_ele_order):
    eleVec=[]
    xg=[]
    for i in range(len(elelst)):
        xa=nodelst[elelst[i,0]-1]
        xb=nodelst[elelst[i,1]-1]
        xc=nodelst[elelst[i,2]-1]

        # if(i==12):
        #     plt.scatter(xa[0],xa[1],color='r')  #查看节点逆时针还是顺时针
        #     plt.scatter(xb[0],xb[1],color='b')
        #     plt.scatter(xc[0],xc[1],color='y')
        #     plt.show()

        xg.append([np.mean([xa[0],xb[0],xc[0]]),np.mean([xa[1],xb[1],xc[1]]),np.mean([xa[2],xb[2],xc[2]])])

        vba=xb-xa
        vca=xc-xa

       
        
        #jud_ele_order=True
        if(jud_ele_order==True): 
            #节点顺时针ac*ab
            ev31 = vca[1]*vba[2]-vca[2]*vba[1]
            ev32 = vca[2]*vba[0]-vca[0]*vba[2]
            ev33 = vca[0]*vba[1]-vca[1]*vba[0]
        else:
            #节点逆时针WMF
            ev31 = vba[1]*vca[2]-vba[2]*vca[1]
            ev32 = vba[2]*vca[0]-vba[0]*vca[2]
            ev33 = vba[0]*vca[1]-vba[1]*vca[0]
        rr = math.sqrt(ev31*ev31+ev32*ev32+ev33*ev33)
        # unit vectors for local coordinates of elements
        ev31 /=rr
        ev32 /=rr
        ev33 /= rr

        if( abs(ev33) < 1 ):
            ev11 = ev32
            ev12 = -ev31
            ev13 = 0 
            rr = math.sqrt(ev11*ev11 + ev12*ev12) 
            ev11 /=rr
            ev12 /=rr
        
        else:
            ev11= 1
            ev12= 0
            ev13= 0
        

        ev21 = ev32*ev13-ev33*ev12
        ev22 = ev33*ev11-ev31*ev13
        ev23 = ev31*ev12-ev32*ev11
        eleVec.append([ev11,ev12,ev13,ev21,ev22,ev23,ev31,ev32,ev33])
    eleVec=np.array(eleVec)
    xg=np.array(xg)
    #print(eleVec)
    return eleVec,xg



def read_node(fname):
    f=open(fname,'r')
    data0=[]
    data1=[]
    kline=0
    for line in f:
        kline=kline+1
        N1=len(line.split())
        if(N1>0):
            datatem=[]
            for k in range(N1):
                tem=float(line.split()[k])
                datatem.append(tem)
            if(N1==4):
                data0.append(datatem)
            elif(N1==3):
                data1.append(datatem)
    f.close()
    return np.array(data0),np.array(data1)

def read_elenum():
    Fele=[]
    Aele=[]
    Sele=[]
    f=open('indats/in_fgeom.dat')
    for line in f:
        Fele.append(int(line.split()[0]))
    f.close()
    f=open('indats/in_sgeom.dat')
    for line in f:
        Sele.append(int(line.split()[0]))
    f.close()
    # f=open('indats/in_ageom.dat')
    # for line in f:
    #     Aele.append(int(line.split()[0]))
    # f.close()
    return np.array(Fele),np.array(Sele),np.array(Aele)

def read_data(fname,K,N1):
    f=open(fname,'r')
    data1=[]
    kline=0
    for line in f:
        #if(len(line.split())>0 and kline>=K*N1 and kline<(K+1)*N1):
        tem=line.split()[0]
        data1.append(float(tem))
        kline=kline+1
    f.close()
    return np.array(data1)

def read_dats(K,teminfo):
    fv1=read_data('dats/fsnapd1.dat',K,int(teminfo[0]))
    fv2=read_data('dats/fsnapd2.dat',K,int(teminfo[0]))
    fs1=read_data('dats/fsnaps1.dat',K,int(teminfo[0]))
    fs2=read_data('dats/fsnaps2.dat',K,int(teminfo[0]))
    ft1=read_data('dats/fsnapt1.dat',K,int(teminfo[0]))
    ft2=read_data('dats/fsnapt2.dat',K,int(teminfo[0]))
    ft3=read_data('dats/fsnapt3.dat',K,int(teminfo[0]))
    arriT=read_data('dats/arriveT.dat',K,int(teminfo[0]))
    
    sv1=read_data('dats/ssnapd1.dat',K,int(teminfo[1]+teminfo[2]))
    sv2=read_data('dats/ssnapd2.dat',K,int(teminfo[1]+teminfo[2]))
    sv3=read_data('dats/ssnapd3.dat',K,int(teminfo[1]+teminfo[2]))
    ss1=read_data('dats/ssnaps1.dat',K,int(teminfo[1]+teminfo[2]))
    ss2=read_data('dats/ssnaps2.dat',K,int(teminfo[1]+teminfo[2]))
    ss3=read_data('dats/ssnaps3.dat',K,int(teminfo[1]+teminfo[2]))
    return fv1,fv2,fs1,fs2,ft1,ft2,ft3,sv1,sv2,sv3,ss1,ss2,ss3,arriT

def trans(Fele,Sele,Aele,fv,sv):
    N=len(Fele)+len(Sele)+len(Aele)
    data1=[0]*N

    for i in range(len(fv)):
        data1[Fele[i]]=fv[i]
    for i in range(len(sv)):
        if(i<len(Sele)):
            data1[Sele[i]]=sv[i]
        else:
            data1[Aele[i-len(Sele)]]=sv[i]
    return np.array(data1)

# %%

"""
Get input values and create mesh
"""

# =====================
# Parameters
# =====================
'''
Input parameters
1. location of folder
2. Length of the fault
3. tectonic plate rate 
'''

target_folder = sys.argv[1]+'/pars'
L1 = float(sys.argv[2])
V_pl = float(sys.argv[3])

os.makedirs(target_folder, exist_ok=True)
os.chdir(target_folder)


W1 = 40E3     # Fault 1 length, width


MU = 32038120320.0  # Lame Constant
sigma_clip = 1e8  # Maximum effective normal stress
b = 0.015            # State evolution parameter
a_min_b = 0.01      # RSF par 
kx = 10 #Length/Nucleation length ratio kx=L/L_nuc (Rubin&Ampuero,2005) 
dc = W1/2 * sigma_clip * b * math.pi / MU / kx * (b/a_min_b)**-2
print(f'dc : {dc:2.5f}')


# Nucleation length Lb (Dieterich, 1992)
Lb = MU*dc/sigma_clip/b

## Minimum edge is defined by the length scale Lb
resolution = 7
DX = Lb/resolution ## edge size

## Minimum edge is defined by the length scale Lb
strike1, dip1 = 90, 90   # Pure strike slip, strike align on the x axis


# Depth points
Z1 = W1 * math.sin(dip1*math.pi/180) + 2000

# First points
x01, y01, z01 = 0, 0, -Z1


# # =====================
# # Gmsh model
# # =====================
gmsh.initialize()
gmsh.model.add("geometry")

faults = []
for i, (x0, y0, z0, L, W, strike, dip) in enumerate([
    (x01, y01, z01, L1, W1, strike1, dip1)
]):
    corners = fault_corners(x0, y0, z0, L, W, strike, dip)
    points = [gmsh.model.geo.addPoint(*p,DX) for p in corners]

    l1 = gmsh.model.geo.addLine(points[0], points[1])
    l2 = gmsh.model.geo.addLine(points[1], points[2])
    l3 = gmsh.model.geo.addLine(points[2], points[3])
    l4 = gmsh.model.geo.addLine(points[3], points[0])

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    faults.append(s)

# Synchronize and generate mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)

# Save mesh
gmsh.option.setNumber("Mesh.MshFileVersion", 2)
gmsh.write("geometry.msh")



# %%
'''
Change rate and state parameters
'''

# Read the geomatry
fnamegeo='geometry.msh'

nodelst,elelst=read_mshV2(fnamegeo)
eleVec,xg=get_eleVec(nodelst,elelst,False)
#sprint(xg)
nodelst[:,2]=nodelst[:,2]-np.max(nodelst[:,2])

angle_rad = strike1 * np.pi/180.0
Slip_plate=[math.cos(angle_rad),math.sin(angle_rad),0]

# initialize the parameters 
Tno=np.zeros(elelst.shape[0])
depele=np.zeros(elelst.shape[0])
alst=np.zeros(elelst.shape[0])
blst=np.zeros(elelst.shape[0])
rake=np.zeros(elelst.shape[0])
Tt=np.zeros(elelst.shape[0])
Tt1o=np.zeros(elelst.shape[0])
Tt2o=np.zeros(elelst.shape[0])
Dc=np.zeros(elelst.shape[0])

## Depth dependent variation of parameters
## Effective normal stress change with depth
zz = np.arange(-0,-45,-1)*1e3
cp = (-18e3*zz + 1e6)*1e-6
cp = np.clip(cp, a_min=None, a_max=sigma_clip*1e-6) 
f_depth_sigma = interp1d(zz, cp, kind='linear', fill_value='extrapolate')


L_edge = 5e3
avary_depth=np.array([[0,b+a_min_b],[-5e3,b], [-10e3,b-a_min_b],[-20e3,b-a_min_b],[-25e3,b],[-40e3,b+a_min_b*2]])
avary_strike=np.array([[0,b+a_min_b],[L_edge,b-a_min_b], [L1-L_edge,b-a_min_b], [L1,b+a_min_b]])

fa_depth = interp1d(avary_depth[:,0], avary_depth[:,1], kind='linear', fill_value='extrapolate')
fa_strike = interp1d(avary_strike[:,0], avary_strike[:,1], kind='linear', fill_value='extrapolate')

f0=0.6
slipv=V_pl
V0=1e-6
slipvC=1.1 * V_pl
for i in range(len(elelst)):
    #print(xg[i,2])
    ev11,ev12,ev13=eleVec[i,0],eleVec[i,1],eleVec[i,2]
    ev21,ev22,ev23=eleVec[i,3],eleVec[i,4],eleVec[i,5]
    Tt1=Slip_plate[0]*ev11+Slip_plate[1]*ev12+Slip_plate[2]*ev13
    Tt2=Slip_plate[0]*ev21+Slip_plate[1]*ev22+Slip_plate[2]*ev23
    rake[i]=0

    temlst=xg[i,2]     # Depth in km
    Tno[i]=f_depth_sigma(temlst)

    alst[i]=max(fa_depth(temlst), fa_strike(xg[i,0]))
        
    blst[i]=b
    Tt[i]=Tno[i]*alst[i]*arcsinh(slipv/(2.0*V0)* math.exp((f0+blst[i]*math.log(V0/slipvC))/alst[i]))
    #Tt[i]=Tno[i]*0.85        
    Tt1o[i]=Tt[i]*math.cos(rake[i])
    Tt2o[i]=Tt[i]*math.sin(rake[i])
    Dc[i]=dc
    
    
# ouputVTK('cascadia.vtk')
print(rake.shape[0])
f=open('init.txt','w')
for i in range(rake.shape[0]):
    f.write('%f %f %f %f %f %f %f %E 0.0 0.0\n' %(rake[i],alst[i],blst[i],Dc[i],f0,Tt[i],Tno[i],V_pl))
f.close()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)


ax.set_xlabel('x [km]')
ax.set_ylabel('z [km]')
# ax.set_zlabel('z')

sctr= ax.scatter(xg[:,0]*1e-3, xg[:,2]*1e-3, c = np.array(alst) - np.array(blst),s =1)
fig.colorbar(sctr, ax=ax, shrink=0.3, label="a-b", 
             # orientation = 'horizontal',
             # location='top'
                     )

fig.savefig('fric_setup.jpg')


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# ax.view_init(elev=30, azim=10, roll=0) 
# ax.set_box_aspect([0.1, 3, 0.75]) 

ax.set_xlabel('x [km]')
ax.set_ylabel('z [km]')
# ax.set_zlabel('z')

sctr= ax.scatter(xg[:,0]*1e-3, xg[:,2]*1e-3, c = Tno,s =1)
fig.colorbar(sctr, ax=ax, shrink=0.3, label="$\\sigma_n$", 
             # orientation = 'horizontal',
             # location='top'
                     )

fig.savefig('Tno_setup.jpg')


hRA=2.0/np.pi*MU*b*dc/a_min_b**2/sigma_clip
h=hRA*np.pi*np.pi/4.0
hRR=np.pi/4.0*MU*dc/(a_min_b)/sigma_clip

A0=9.0*np.pi/32*MU*dc/(b*sigma_clip)

print('hRA',hRA,)
print('hRR',hRR,)
print('h',h,)
print('Cohesive zone:',A0)



