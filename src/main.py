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



file_name = sys.argv[0]
print(file_name)

from mpi_config import comm, rank, size

import sys
#old_stdout = sys.stdout
#log_file = open("message.log", "w")
#sys.stdout = log_file

if __name__ == "__main__":
    #eleVec,xg=None,None
    #nodelst,xg=None,None
    sim0=None
    jud_coredir=None
    blocks_to_process=[]
    if(rank==0):
        print('# ----------------------------------------------------------------------------\n\
# PyQuake3D: Boundary Element Method to simulate sequences of earthquakes and aseismic slips\n\
# * 3D non-planar quasi-dynamic earthquake cycle simulations\n\
# * Support for Hierarchical matrix compressed storage and calculation\n\
# * Parallelized with MPI (%d cpus)\n\
# * Support for rate-and-state aging friction laws\n\
# * Supports output to VTK formats\n\
# * ----------------------------------------------------------------------------'\
            %size)
        try:
            #start_time = time.time()
            parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
            parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
            parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

            args = parser.parse_args()

            fnamegeo = args.inputgeo
            fnamePara = args.inputpara
        
        except:
            #fnamegeo='examples/HF-model/planar10w.msh'
            #fnamePara='examples/HF-model/parameter.txt'
            #fnamegeo='examples/BP5-QD/bp5t.msh'
            #fnamePara='examples/BP5-QD/parameter.txt'
            #fnamegeo='examples/EAFZ-model/turkey_cut.msh'
            #fnamePara='examples/EAFZ-model/parameter.txt'
            #fnamegeo='examples/WMF/WMF3.msh'
            #fnamePara='examples/WMF/parameter.txt'
            #fnamegeo='examples/cascadia/cascadia35km_ele4.msh'
            #fnamePara='examples/cascadia/parameter.txt'
            fnamegeo='examples/Lab-model/lab.msh'
            fnamePara='examples/Lab-model/parameter.txt'

        
            
            
        print('Input msh geometry file:',fnamegeo, flush=True)
        print('Input parameter file:',fnamePara, flush=True)   

        nodelst,elelst=readmsh.read_mshV2(fnamegeo)
        sim0=QDsim.QDsim(elelst,nodelst,fnamePara)

        print('Number of Node',nodelst.shape, flush=True)
        print('Number of Element',elelst.shape, flush=True)
        
        

        

        f=open('state.txt','w')
        f.write('Program start time: %s\n'%str(datetime.now()))
        f.write('Input msh geometry file:%s\n'%fnamegeo)
        f.write('Input parameter file:%s\n'%fnamePara)
        f.write('Number of Node:%d\n'%nodelst.shape[0])
        f.write('Number of Element:%d\n'%elelst.shape[0])
        f.write('Cs:%f\n'%sim0.Cs)
        f.write('First Lamé constants:%f\n'%sim0.lambda_)
        f.write('Shear Modulus:%f\n'%sim0.mu)
        
        f.write('Youngs Modulus:%f\n'%sim0.YoungsM)
        f.write('Poissons ratio:%f\n'%sim0.possonratio)
        f.write('maximum element size:%f\n'%sim0.maxsize)
        f.write('average elesize:%f\n'%sim0.ave_elesize)
        f.write('Critical nucleation size:%f\n'%sim0.hRA)
        f.write('Cohesive zone::%f\n'%sim0.A0)
        f.write('iteration time_step(s) maximum_slip_rate(m/s) time(s) time(h)\n')


        jud_coredir,blocks_to_process=sim0.get_block_core()
        print('jud_coredir',jud_coredir) #if saved corefunc
        if(jud_coredir==False):
            print('Start to calculate Hmatrix...')
        else:
            print('Hmatrix reading...')
        SLIPV=[]
        Tt=[]
        #test green functions
        # x=np.ones(len(elelst))
        # start_time = time.time()
        # for i in range(1):
        #     y=sim0.tree_block.blocks_process_MVM(x,blocks_to_process,'A1d')
        #     print(y[:20])
        # end_time = time.time()
        # print(f"Green func calc_MVM_fromC Time taken: {end_time - start_time:.10f} seconds")

        
        

        # initial condition loaded by previous results
        #fname="step17800.vtk"
        #sim0.read_vtk(fname)

        fname='Init.vtk'
        sim0.ouputVTK(fname)


    
    print('rank:',rank)
   
    sim0 = comm.bcast(sim0, root=0)
    jud_coredir = comm.bcast(jud_coredir, root=0)

    
    
    if(jud_coredir==False):
        #sim0.local_blocks=sim0.tree_block.parallel_traverse_SVD(sim0.Para0['Corefunc directory'],plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
        if(rank==0):
           sim0.tree_block.master(sim0.Para0['Corefunc directory'],blocks_to_process,size-1,save_corefunc=sim0.save_corefunc)
        else:
           sim0.tree_block.worker()
        #sim0.tree_block.master_scatter(sim0.Para0['Corefunc directory'],blocks_to_process,size)
        sim0.local_blocks=sim0.tree_block.parallel_block_scatter_send(sim0.tree_block.blocks_to_process,plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
    else:
        sim0.local_blocks=sim0.tree_block.parallel_block_scatter_send(blocks_to_process,plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
        


    
    
        

    start_time = MPI.Wtime()
    totaloutputsteps=int(sim0.Para0['totaloutputsteps'])
    for i in range(totaloutputsteps):
    #for i in range(0):

        if(i==0):
            dttry=sim0.htry
        else:
            dttry=dtnext
        dttry,dtnext=sim0.simu_forward(dttry)
        #sim0.simu_forward(dttry)
        if(rank==0):
            year=sim0.time/3600/24/365
            if(i%20==0):
                print('iteration:',i, flush=True)
                print('dt:',dttry,' max_vel:',np.max(np.abs(sim0.slipv)),' Seconds:',sim0.time,'  Days:',sim0.time/3600/24,
                'year',year, flush=True)
            f.write('%d %f %f %.16e %f %f\n' %(i,dttry,np.max(np.abs(sim0.slipv1)),np.max(np.abs(sim0.slipv2)),sim0.time,sim0.time/3600.0/24.0))
            
            #f1.write('%d %f %f %f %.6e %.16e\n'%(i,dttry,sim0.time,sim0.time/3600.0/24.0,sim0.Tt[index1_],sim0.slipv[index1_]))
            #SLIP.append(sim0.slip)
            SLIPV.append(sim0.slipv)
            Tt.append(sim0.Tt)
            
            # if(sim0.time>60):
            #     break
            outsteps=int(sim0.Para0['outsteps'])
            directory='out_vtk'
            if not os.path.exists(directory):
                os.mkdir(directory)
            if(i%outsteps==0):
                #SLIP=np.array(SLIP)
                SLIPV=np.array(SLIPV)
                Tt=np.array(Tt)
                if(sim0.Para0['outputSLIPV']=='True'):
                    directory1='out_slipvTt'
                    if not os.path.exists(directory1):
                        os.mkdir(directory1)
                    np.save(directory1+'/slipv_%d'%i,SLIPV)
                if(sim0.Para0['outputTt']=='True'):
                    directory1='out_slipvTt'
                    if not os.path.exists(directory1):
                        os.mkdir(directory1)
                    np.save(directory1+'/Tt_%d'%i,Tt)
                #np.save('examples/Heto/slip/slip_%d'%i,SLIP)
                #np.save('examples/Heto/Tt/Tt_%d'%i,Tt)

                #SLIP=[]
                SLIPV=[]
                Tt=[]

                if(sim0.Para0['outputvtk']=='True'):
                    #print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                    fname=directory+'/step'+str(i)+'.vtk'
                    sim0.ouputVTK(fname)
                if(sim0.Para0['outputmatrix']=='True'):
                    fname='step'+str(i)
                    sim0.outputtxt(fname)
                #if(year>1200 or np.max(np.abs(sim0.slipv))<0.01):
                #if(year>280):
                #    break
            



    end_time = MPI.Wtime()

    if rank == 0:
        
        print(f"Program run time: {end_time - start_time:.6f} sec")
        timetake=end_time - start_time
        f.write('Program end time: %s\n'%str(datetime.now()))
        f.write("Time taken: %.2f seconds\n"%timetake)
        f.close()
        #print('menmorary:',s*6)
        
#sys.stdout = old_stdout
#log_file.close()







