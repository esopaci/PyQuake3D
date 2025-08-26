import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import SH_greenfunction
import DH_greenfunction
import os
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# global Para
# Para={}

def readPara0(fname):
    global Para0
    Para0={}
    f=open(fname,'r')
    for line in f:
        #print(line)
        if ':' in line:
            tem=line.split(':')
            #if(tem[0]==)
            Para0[tem[0].strip()]=tem[1].strip()




# def load_config(data_dir):
#     global Para
#     Para={}
#     Para0=readPara0(data_dir)
#     Para['Corefunc directory']=Para0['Corefunc directory']

#     Para['save Corefunc']=Para0['save Corefunc']=='True'
#     Para['Node_order']=Para0['Node_order']=='True'
#     Para['Scale_km']=Para0['Scale_km']=='True'
#     Para['Shear modulus']=float(Para0['Shear modulus'])
#     Para['Lame constants']=float(Para0['Lame constants'])
#     Para['Rock density']=float(Para0['Rock density'])
#     Para['Half space']=Para0['Half space']=='True'
#     Para['InputHetoparamter']=Para0['InputHetoparamter']=='True'
#     Para['Hmatrix_mpi_plot']=Para0['Hmatrix_mpi_plot']=='True'

#     Para['If Dilatancy']=Para0['If Dilatancy']=='True'
#     Para['DilatancyC']=float(Para0['Dilatancy coefficient'])
#     Para['Hydraulic diffusivity']=float(Para0['Hydraulic diffusivity'])
#     #self.hw=float(self.Para0['Low permeability zone thickness'])
#     Para['Actively shearing zone thickness']=float(Para0['Actively shearing zone thickness'])
#     Para['Effective compressibility']=float(Para0['Effective compressibility'])
#     Para['GPU']=Para0['GPU']=='True'
#     Para['Using C++ green function']=Para0['Using C++ green function']=='True'


    # Para['ssv_scale']=float(Para0['Vertical principal stress'])
    # Para['ssh1_scale']=float(Para0['Maximum horizontal principal stress'])
    # Para['ssv0_scale']=float(Para0['Minimum horizontal principal stress'])
    # Para['trac_nor']=float(Para0['Vertical principal stress value'])
    # Para['Constant porepressure']=float(Para0['Constant porepressure'])
    # Para['Initial porepressure']=float(Para0['Initial porepressure'])


    # Para['Vertical principal stress value varies with depth']=Para0['Vertical principal stress value varies with depth']=='True'


