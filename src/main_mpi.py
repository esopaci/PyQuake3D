import readmsh
import numpy as np
import sys
import matplotlib.pyplot as plt
import QDsim
from math import *
import time
import argparse
import os
import psutil
from datetime import datetime
from mpi4py import MPI
import config

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

file_name = sys.argv[0]
print(file_name)

from config import comm, rank, size

import sys
#old_stdout = sys.stdout
#log_file = open("message.log", "w")
#sys.stdout = log_file

if __name__ == "__main__":
    #eleVec,xg=None,None
    #nodelst,xg=None,None
    sim0=None
    fnamePara=None
    # jud_coredir=None
    blocks_to_process=[] #Save block submatirx from Hmatrix
    if(rank==0):
        print('# ----------------------------------------------------------------------------')
        print('# PyQuake3D: Boundary Element Method to simulate sequences of earthquakes and aseismic slips')
        print('# * 3D non-planar quasi-dynamic earthquake cycle simulations')
        print('# * Support for Hierarchical matrix compressed storage and calculation')
        print(f'# * Parallelized with MPI ({size} cpus)')
        print('# * Support for rate-and-state aging friction laws')
        print('# * Supports output to VTU formats')
        print('# * ----------------------------------------------------------------------------')
        try:
            #start_time = time.time()
            parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
            parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
            parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

            args = parser.parse_args()

            fnamegeo = args.inputgeo
            fnamePara = args.inputpara
        
        except:
            # fnamegeo='examples/EAFZ-model/turkey.msh'
            # fnamePara='examples/EAFZ-model/parameter.txt'
            fnamegeo='examples/BP5-QD/bp5t.msh'
            fnamePara='examples/BP5-QD/parameter.txt'
            # fnamegeo='examples/cascadia/50km_43dense_35w.msh'
            # fnamePara='examples/cascadia/parameter.txt'
        
            
        print('Input msh geometry file:',fnamegeo, flush=True)
        print('Input parameter file:',fnamePara, flush=True)   

        nodelst,elelst=readmsh.read_mshV2(fnamegeo)
        Para=config.readPara(fnamePara)
        sim0=QDsim.QDsim(elelst,nodelst,Para)
    #     # output intial results
        fname='Init.vtu'
        sim0.writeVTU(fname,init=True)
        #print(sim0.P)

    sim0 = comm.bcast(sim0, root=0)
    sim0.calc_greenfuncs_mpi()
    sim0.start()
    





