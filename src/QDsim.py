import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import SH_greenfunction
import DH_greenfunction
import os
import sys
#import json
from concurrent.futures import ProcessPoolExecutor
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata
import readmsh
#import cupy as cp
from collections import deque
from scipy.ndimage import gaussian_filter1d
import Hmatrix as Hmat
import joblib
from config import comm, rank, size
from mpi4py import MPI
import pyvista as pv
from scipy.linalg import lu_factor, lu_solve

def get_sumS(X,Y,Z,nodelst,elelst):
    Ts,Ss,Ds=0,0,1
    mu=0.33e11
    lambda_=0.33e11
    Strs=[]
    Stra=[]
    Dis=[]
    for i in range(len(elelst)):
        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda_)

        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        ue,un,uv=DH_greenfunction.TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,0.25)
        
        Dis_tems=np.array([ue,un,uv])
        #print(ue.shape,un.shape)
        if(len(Strs)==0):
            Strs=Stress
            Stra=Strain
            Dis=Dis_tems
        else:
            Strs=Strs+Stress
            Stra=Stra+Strain
            Dis=Dis+Dis_tems
    return Dis,Strs,Stra

# find the mesh boundary_edges and nodes
def find_boundary_edges_and_nodes(triangles):
    from collections import defaultdict
    edge_count = defaultdict(int)
    boundary_nodes = set()

    # 遍历每个三角形，统计边的出现次数
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # 找到只出现一次的边，并记录边上的节点
    boundary_edges = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_edges.append(edge)
            boundary_nodes.update(edge)

    return boundary_edges, np.array(list(boundary_nodes))

from scipy.spatial.distance import cdist

# Calculate the distance between two node coord
def find_min_euclidean_distance(coords1, coords2):
    # 使用 scipy.spatial.distance.cdist 计算成对距离
    distances = cdist(coords1, coords2, 'euclidean')
    # 找到最小距离及其对应的索引
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_idx]
    return min_distance
# Read parameters
def readPara0(fname):
    Para0={}
    f=open(fname,'r')
    for line in f:
        #print(line)
        if ':' in line:
            tem=line.split(':')
            #if(tem[0]==)
            Para0[tem[0].strip()]=tem[1].strip()
    return Para0

def clac_revn_vchange_filter(v):
    sigma = 2
    smoothed_y = gaussian_filter1d(v, sigma=sigma)
    revn=[]
    vchange=[]
    for i in range(len(v)):
        vchange.append(abs((v[i]-smoothed_y[i])/smoothed_y[i]))
    #plt.plot(v)
    #plt.plot(smoothed_y)
    #vchange=np.array(vchange)*(1.0-exp(-10000*np.mean(v)))
    if(np.mean(v)>0.001):
        vchange=np.array(vchange)*0.0
    #plt.show()
    return np.mean(vchange)




class QDsim:
    def __init__(self,elelst,nodelst,fnamePara):
        #for i in range(len(xg)):
        

        last_backslash_index = fnamePara.rfind('/')

        # get Parameter file name
        if last_backslash_index != -1:
            self.dirname = fnamePara[:last_backslash_index]
        else:
            self.dirname = fnamePara
        #print(self.dirname)
        self.Para0=readPara0(fnamePara)
        #parameter define
        self.Corefunc_directory=self.Para0['Corefunc directory']
        self.save_corefunc=self.Para0['save Corefunc']=='True'
        jud_ele_order=self.Para0['Node_order']=='True'
        jud_scalekm=self.Para0['Scale_km']=='True'
        self.mu=float(self.Para0['Shear modulus'])
        self.lambda_=float(self.Para0['Lame constants'])
        self.density=float(self.Para0['Rock density'])
        self.halfspace_jud=self.Para0['Half space']=='True'
        self.InputHetoparamter=self.Para0['InputHetoparamter']=='True'
        self.hmatrix_mpi_plot=self.Para0['Hmatrix_mpi_plot']=='True'

        self.Ifdila=self.Para0['If Dilatancy']=='True'
        self.DilatancyC=float(self.Para0['Dilatancy coefficient'])
        self.Chyd=float(self.Para0['Hydraulic diffusivity'])
        #self.hw=float(self.Para0['Low permeability zone thickness'])
        self.hs=float(self.Para0['Actively shearing zone thickness'])
        self.EPermeability=float(self.Para0['Effective compressibility'])
        #self.useGPU=self.Para0['GPU']=='True'
        self.useC=self.Para0['Using C++ green function']=='False'
        
        #self.tf=2.0*self.Chyd/self.hs/self.hw
        

        
        if(jud_scalekm==False):
            nodelst=nodelst/1e3
        #jud_ele_order=False
        # get element label and element center coodinate
        eleVec,xg=readmsh.get_eleVec(nodelst,elelst,jud_ele_order)
        self.eleVec=eleVec
        self.elelst=elelst
        self.nodelst=nodelst
        self.xg=np.array(xg, dtype=np.float64)

        self.maxslipvque=deque(maxlen=20)
        self.val_nv=0
        
        

        
        self.htry=1e-3
        self.Cs=sqrt(self.mu/self.density)
        self.time=0
        
        #self.useGPU=self.Para0['GPU']=='True'
        #self.num_process=int(self.Para0['Processors'])
        #self.Batch_size=int(self.Para0['Batch_size'])
        self.YoungsM=self.mu*(3.0*self.lambda_+2.0*self.mu)/(self.lambda_+self.mu)
        self.possonratio=self.lambda_/2.0/(self.lambda_+self.mu)

        print('Cs',self.Cs)
        print('First Lamé constants',self.lambda_)
        print('Shear Modulus',self.mu)
        print('Youngs Modulus',self.YoungsM)
        print('Poissons ratio',self.possonratio)
        
        self.Init_condition()

        
        #self.calc_corefunc()
        #Calcultae Hmatrix structure
        self.tree_block=Hmat.createHmatrix(self.xg,self.nodelst,self.elelst,self.eleVec,self.mu,self.lambda_,self.halfspace_jud,plotHmatrix=True)

        

        # f=open('Tvalue.txt','w')
        # f.write('xg1,xg2,xg3,se1,se2,se3\n')
        # for i in range(len(xg)):
        #     #f.write('%f %f %f %f %f %f\n' %(xg[i,0],xg[i,1],xg[i,2],self.Tt1o[i],self.Tt2o[i],self.Tno[i]))
        #     f.write('%f,%f,%f,%f,%f,%f\n' %(xg[i,0],xg[i,1],xg[i,2],self.T_globalarr[i,0],self.T_globalarr[i,1],self.T_globalarr[i,2]))
        # f.close()
    


    #Determine whether Hmatrix has been calculated. If it has been calculated, read it directly
    def get_block_core(self):
        jud_coredir=True
        #directory = 'surface_core'
        
        if os.path.exists(self.Corefunc_directory):
            file_path = os.path.join(self.Corefunc_directory, 'blocks_to_process.joblib')
            if not os.path.exists(file_path):
                jud_coredir=False
        else:
            os.mkdir(self.Corefunc_directory)
            jud_coredir=False
        blocks_to_process=[]
        if(jud_coredir==True):
            blocks_to_process = joblib.load(self.Corefunc_directory+'/blocks_to_process.joblib')
        #make sure all submatices are loaded
        self.blocks_to_process=blocks_to_process
        for i in range(len(blocks_to_process)):
            if(hasattr(blocks_to_process[i], 'judproc') and blocks_to_process[i].judproc==False):
                jud_coredir=False
                break
        return jud_coredir,blocks_to_process
        # elif():
        #     self.tree_block.parallel_traverse_SVD(comm, rank, size)



    def get_rotation1(self,x):
        if(x<70):
            theta=-10
        elif(x>=60.0 and x<80.0):
            temx=((x-60.0)/10.0-1.0)*np.pi/2
            theta=-10.0-(sin(temx)+1.0)*10.0
        else:
            theta=-30.0
        return theta

    

    def randompatch(self):
        xmin, xmax = -22, 22
        ymin, ymax = -28, -18

        # 生成 N 个随机点
        N = 25
        np.random.seed(42)
        x_random = np.random.uniform(xmin, xmax, N)*1e3
        y_random = np.random.uniform(ymin, ymax, N)*1e3
        sizeR_random=np.random.uniform(0.7, 2.0, N)*1e3

        for i in range(N):
            nuclearloc=[x_random[i],0,y_random[i]]
            distem=np.linalg.norm(self.xg-nuclearloc,axis=1)
            index1=np.where(distem<sizeR_random[i])[0]
            #print(len(index1),sizeR_random[i])
            self.a[index1]=0.01
            self.b[index1]=0.025
            self.dc[index1]=0.015
            #print(nuclearloc)
    
    #calc_nucleaszie and cohesivezone
    def calc_nucleaszie_cohesivezone(self):
        maxsize=0
        elesize=[]
        for i in range(len(self.eleVec)):
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            sizeA=np.linalg.norm(P1-P2)
            sizeB=np.linalg.norm(P1-P3)
            sizeC=np.linalg.norm(P2-P3)
            size0=np.max([sizeA,sizeB,sizeC])
            if(size0>maxsize):
                maxsize=size0
            elesize.append(size0)
        elesize=np.array(elesize)
        self.maxsize=maxsize
        self.ave_elesize=np.mean(elesize)
        b=np.max(self.b)
        a=np.min(self.a)
        #b=0.024
        #a=0.0185
        #b=0.025
        #a=0.01
        sigma=np.mean(self.Tno*1e6)
        #print(self.Tno)
        L=np.min(self.dc)
        #L=0.015
        #print('L:',L)
        print('a,b,L:',a,b,L)
        self.hRA=2.0/np.pi*self.mu*b*L/(b-a)/(b-a)/sigma
        self.hRA=self.hRA*np.pi*np.pi/4.0
        self.A0=9.0*np.pi/32*self.mu*L/(b*sigma)
        
        print('maximum element size',maxsize, flush=True)
        print('average elesize',self.ave_elesize, flush=True)
        print('Critical nucleation size',self.hRA, flush=True)
        print('Cohesive zone:',self.A0, flush=True)
        return maxsize
    
    #set heterogeneous plate slip rate
    def Grad_slpv_con(self,const):
        self.slipvC=np.ones(len(self.xg))*self.Vpl_con
        Vpl_min=1e-16
        if(const==False):
            for i in range(len(self.xg)):
                if(self.xg[i,2]<-5000):
                    self.slipvC[i]=self.Vpl_con
                else:
                    self.slipvC[i]=Vpl_min+abs(self.xg[i,2])/5000.0*(self.Vpl_con-Vpl_min)

    #set heterogeneous normal stress
    def Tn_edge(self):
        np.random.seed(42) 
        #self.Tno[i]=self.Tno[i]*exp(dis1)
        boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(self.elelst)
        boundary_coord=self.nodelst[boundary_nodes-1]
        index_surface=np.where(np.abs(boundary_coord[:,2]-0.0)<1e-5)[0]
        index_b=np.arange(0,len(boundary_coord),1)
        index_sb=np.setdiff1d(index_b,index_surface)
        boundary_coord_sb=boundary_coord[index_sb]
        Wedge=0.1*2.0*1e-2
        edTno=self.Tno[0]*0.7
        maxTno=self.Tno[0]
        Tno1=np.copy(self.Tno)
        index1=[]
        index0=[]
        for i in range(len(self.xg)):
            coords1=np.array([self.xg[i]])
            #print(coords1.shape, boundary_coord.shape)
            distem=find_min_euclidean_distance(coords1, boundary_coord_sb)
            dis=distem/Wedge
            
            if(dis<1.0):
                #self.a[i]=aVs-(aVs-aVw)*dis1
                self.Tno[i]=edTno-(edTno-maxTno)*dis
            else:
                index0.append(i)
                
        #print('index1',len(index1))
        index1=np.arange(0,len(self.Tno),1)
        Ns=len(self.Tno)
        sample = np.random.choice(index1, size=Ns, replace=False)
        #arr = np.random.normal(loc=0.0, scale=0.3, size=100)
        arr = np.random.uniform(low=-0.5, high=0.5, size=Ns)
        for i in range(Ns):
            j=sample[i]
            Tno1[j]=self.Tno[j]+self.Tno[j]*arr[i]
        
        Xmin=np.min(self.xg[:,0])
        Xmax=np.max(self.xg[:,0])
        Y=self.xg[:,1]
        #y1=np.max(self.xg[:,1])-self.xg[:,1]
        #Y=np.stack((y1, self.xg[:,2]), axis=1) 
        #Y = np.linalg.norm(Y, axis=1)
        Ymin=np.min(Y)
        Ymax=np.max(Y)
        xi = np.linspace(Xmin, Xmax, 100)
        yi = np.linspace(Ymin, Ymax, 100)
        #print(Xmin, Xmax,Ymin, Ymax)
        Xi, Yi = np.meshgrid(xi, yi)
        

        # 插值结果是一个二维数组
        Si = griddata((self.xg[:,0], self.xg[:,1]), Tno1, (Xi, Yi), method='nearest')  # <<< 这就是 Si

        # 之后可用于滤波
        from scipy.ndimage import gaussian_filter
        smooth_full = gaussian_filter(Si, sigma=4.0)
        #print('smooth_full',np.max(smooth_full))
        # plt.pcolor(smooth_full)
        # plt.savefig('1.png')
        # plt.show()
        
        #pt = np.array([[self.xg[:,0], self.xg[:,1]]])
        points=np.stack((Xi.flatten(), Yi.flatten()), axis=1) 
        #print(points.flatten())
        val = griddata(points, smooth_full.flatten(), self.xg[:,[0,1]], method='nearest')
        nan_indices = np.where(np.isnan(val))[0]
        val[nan_indices]=self.Tno[nan_indices]
        self.Tno[index0]=val[index0]
        
        #print('self.Tno',np.max(self.Tno),self.xg.shape,self.Tno)

        
        



    #set initial condition
    def Init_condition(self):
        N=len(self.eleVec)
        self.Relerrormax1_last=0
        self.Relerrormax2_last=0
        self.Tt1o=np.zeros(N)
        self.Tt2o=np.zeros(N)
        self.Tno=np.zeros(N)
        self.fix_Tn=self.Para0['Fix_Tn']=='True'
        ssv_scale=float(self.Para0['Vertical principal stress'])
        ssh1_scale=float(self.Para0['Maximum horizontal principal stress'])
        ssv0_scale=float(self.Para0['Minimum horizontal principal stress'])
        trac_nor=float(self.Para0['Vertical principal stress value'])
        
        
        self.P0=0
        self.P=np.zeros(N)
        self.dPdt0=np.zeros(N)
        self.local_index=np.arange(0,N,1)
        if(self.Ifdila==True):
            
            c=1e-2
            num=60
            self.P0=float(self.Para0['Constant porepressure'])
            self.P=np.ones(N)*float(self.Para0['Initial porepressure'])
            self.yp=np.logspace(start=-3, stop=10, num=num, base=10)
            self.zp=np.log(1+self.yp/c)
            self.Parr=np.ones([N,num])*float(self.Para0['Initial porepressure'])
            self.Pmatrix=np.zeros([num,num])
            self.Parr=self.Parr*1e6
            self.P0=self.P0*1e6
            self.P=self.P*1e6
            self.Calc_Pmatrix()
            
        
        #print('trac_nor',trac_nor)
        for i in range(N):
            if(self.Para0['Vertical principal stress value varies with depth']=='True'):
                turning_dep=float(self.Para0['Turnning depth'])
                ssv= -self.xg[i,2]/turning_dep+0.2
                if(ssv>1.0):
                    ssv=1.0
                #ssv=ssv*1e6
                ssv=trac_nor*ssv*ssv_scale
            else:
                ssv=trac_nor*ssv_scale
            ssh1=-ssv*ssh1_scale
            ssh2=-ssv*ssv0_scale
            ssv=-ssv
            #ssv= -xg3[i]*maxside/5.;
            #Ph1ang=self.get_rotation1(xg[i,0])-10.0
            #Ph1ang=np.pi/180.*Ph1ang
            Ph1ang=float(self.Para0['Angle between ssh1 and X-axis'])
            Ph1ang=np.pi/180.*Ph1ang
            v11=cos(Ph1ang)
            v12=-sin(Ph1ang)
            v21=sin(Ph1ang)
            v22=cos(Ph1ang)
            Rmatrix=np.array([[v11,v12],[v21,v22]])
            Pstress=np.array([[ssh1,0],[0,ssh2]])
            stress=np.dot(np.dot(Rmatrix,Pstress),Rmatrix.transpose())
            stress3D=np.array([[stress[0][0],stress[0][1],0],[stress[1][0],stress[1][1],0],[0,0,ssv]])
            
            #project stress tensor into fault surface
            #Me=self.eleVec[i].reshape([3,3])
            #T_global=np.dot(Me.transpose(),T_local)
            tra=np.dot(stress3D,self.eleVec[i,-3:])
            #print(tra)
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]
            #print('ev11,ev12,ev13 ',ev11,ev12,ev13)
            #print('ev21,ev22,ev23 ',ev21,ev22,ev23)
            self.Tt1o[i]=tra[0]*ev11+tra[1]*ev12+tra[2]*ev13
            self.Tt2o[i]=tra[0]*ev21+tra[1]*ev22+tra[2]*ev23
            self.Tno[i]=tra[0]*ev31+tra[1]*ev32+tra[2]*ev33
            #print(self.Tt1o[i],self.Tt2o[i],self.Tno[i])
            
            solve_normal=self.Para0['Normal traction solved from stress tensor']=='True'
            if(solve_normal==False):
                self.Tno[i]=ssv

            
            
        
        self.Tno=np.abs(self.Tno)
        self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
        tem=self.Tt/self.Tno
        x=self.Tt1o/self.Tt
        y=self.Tt2o/self.Tt
        solve_shear=self.Para0['Shear traction solved from stress tensor']=='True'
        solve_normal=self.Para0['Normal traction solved from stress tensor']=='True'
        if(self.Para0['Rake solved from stress tensor']=='True'):
            self.rake=np.arctan2(y,x)
            print('rake:',self.rake*180.0/np.pi)
        else:
            self.rake=np.ones(len(self.Tt))*float(self.Para0['Fix_rake'])
            self.rake=self.rake/180.0*np.pi
        
        if(solve_normal==False and self.Para0['Vertical principal stress value varies with depth']!='True'):
            self.Tno=np.ones(len(self.Tno))*trac_nor
        
        #self.rake=np.ones(len(x))*35.0/180.0*np.pi
        self.vec_Tra=np.array([x,y]).transpose()
        
        #print(self.Tt1o)
        #print(np.max(tem),np.min(tem))
        if(self.Para0['Initlab']=='True'):
            self.Tn_edge()

        T_globalarr=[]
        N=self.Tt1o.shape[0]
        self.Vpl_con=1e-6
        self.Vpl_con=float(self.Para0['Plate loading rate'])
        self.Grad_slpv_con(const=True)

        self.Vpls=np.zeros(N)
        self.Vpld=np.zeros(N)
        
        self.shear_loadingS=np.zeros(N)
        self.shear_loadingD=np.zeros(N)
        self.shear_loading=np.zeros(N)
        self.normal_loading=np.zeros(N)

        self.V0=float(self.Para0['Reference slip rate'])
        # self.dc=np.ones(N)*0.01
        # self.f0=np.ones(N)*0.4
        self.dc=np.ones(N)*0.02
        self.f0=np.ones(N)*float(self.Para0['Reference friction coefficient'])
        self.a=np.zeros(N)
        self.b=np.ones(N)*0.03

        self.slipv1=np.zeros(N)
        self.slipv2=np.zeros(N)
        #self.slipv=np.ones(N)*self.Vpl_con
        self.slipv=self.slipvC
        self.slip1=np.zeros(N)
        self.slip2=np.zeros(N)
        self.slip=np.zeros(N)

        self.arriT=np.ones(N)*1e9

        
        if(self.InputHetoparamter==True):
            self.read_parameter(self.Para0['Inputparamter file'])
            #print(min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge)
            #self.randompatch()
            self.maxslipv0=np.max(self.slipv)
            self.rake0=np.copy(self.rake)
            self.calc_nucleaszie_cohesivezone()
        else:
            boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(self.elelst)
            boundary_coord=self.nodelst[boundary_nodes-1]
            index_surface=np.where(np.abs(boundary_coord[:,2]-0.0)<1e-5)[0]
            index_b=np.arange(0,len(boundary_coord),1)
            index_sb=np.setdiff1d(index_b,index_surface)
            boundary_coord_surface=boundary_coord[index_surface]
            boundary_coord_sb=boundary_coord[index_sb]
            #print(boundary_coord.shape,boundary_nodes.shape)

            xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
            ymin,ymax=np.min(self.xg[:,1]),np.max(self.xg[:,1])
            zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])

            nux=float(self.Para0['Nuclea_posx'])
            nuy=float(self.Para0['Nuclea_posy'])
            nuz=float(self.Para0['Nuclea_posz'])
            nuclearloc=np.array([nux,nuy,nuz])
            #nuclearloc=np.array([-20000,0,-20000])
            Wedge=float(self.Para0['Widths of VS region'])
            Wedge_surface=float(self.Para0['Widths of surface VS region'])
            self.localTra=np.zeros([N,2])
            transregion=float(self.Para0['Transition region from VS to VW region'])
            aVs=float(self.Para0['Rate-and-state parameters a in VS region'])
            bVs=float(self.Para0['Rate-and-state parameters b in VS region'])
            dcVs=float(self.Para0['Characteristic slip distance in VS region'])
            aVw=float(self.Para0['Rate-and-state parameters a in VW region'])
            bVw=float(self.Para0['Rate-and-state parameters b in VW region'])
            dcVw=float(self.Para0['Characteristic slip distance in VW region'])

            aNu=float(self.Para0['Rate-and-state parameters a in nucleation region'])
            bNu=float(self.Para0['Rate-and-state parameters b in nucleation region'])
            dcNu=float(self.Para0['Characteristic slip distance in nucleation region'])
            slivpNu=float(self.Para0['Initial slip rate in nucleation region'])
            Set_nuclear=self.Para0['Set_nucleation']=='True'
            Radiu_nuclear=float(self.Para0['Radius of nucleation'])
            ChangefriA=self.Para0['ChangefriA']=='True'

            for i in range(self.Tt1o.shape[0]):
                #tem=min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge
                coords1=np.array([self.xg[i]])
                #print(coords1.shape, boundary_coord.shape)
                distem=find_min_euclidean_distance(coords1, boundary_coord_sb)
                dis=distem/Wedge
                dis_surface=np.copy(dis)
                dis1=(distem-Wedge)/transregion
                if(len(boundary_coord_surface)>10):  #in case there is free surface
                    distem_surface=find_min_euclidean_distance(coords1, boundary_coord_surface)
                    dis_surface=distem_surface/Wedge_surface
                    dis1=min(distem-Wedge,distem_surface-Wedge_surface)/transregion
                nuclearregion=1.0-transregion
                

                if(dis<1.0 or dis_surface<1.0):
                    self.a[i]=aVs
                    self.b[i]=bVs
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcVs
                
                elif(dis1<1.0):
                    
                    if(ChangefriA==True):
                        self.a[i]=aVs-(aVs-aVw)*dis1
                        self.b[i]=bVs
                    else:
                        self.a[i]=aVs
                        self.b[i]=bVs-(bVs-bVw)*dis1
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcVs
                    
                
                else:
                    self.a[i]=aVw
                    self.b[i]=bVw
                    self.dc[i]=dcVw
                    #self.slipv[i]=self.Vpl_con
                    # self.Tt1o[i]=self.Tt1o[i]*2
                    # self.Tt2o[i]=self.Tt2o[i]*2
                

                
                
                distem=np.linalg.norm(self.xg[i]-nuclearloc)

                if(distem<Radiu_nuclear and Set_nuclear==True):
                    self.slipv[i]=slivpNu
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcNu
                    self.a[i]=aNu
                    self.b[i]=bNu


                
                T_local=np.zeros(3)
                T_local[0]=cos(self.rake[i])
                T_local[1]=sin(self.rake[i])
                Me=self.eleVec[i].reshape([3,3])
                T_global=np.dot(Me.transpose(),T_local)
                #print(self.Tt1o[i],self.Tt2o[i],T_global)
                T_globalarr.append(T_global)  
            self.T_globalarr=np.array(T_globalarr)
            
            #print(np.min(self.a))
            # self.Tt1o=self.Tt*np.cos(self.rake)
            # self.Tt2o=self.Tt*np.sin(self.rake)
            self.slipv1=self.slipv*np.cos(self.rake)+1e-16
            self.slipv2=self.slipv*np.sin(self.rake)+1e-16

            if(solve_shear==False):
                self.Tt=(self.Tno-self.P*1e-6)*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
                
                #self.Tt=self.Tt*0.1
                #self.Tt1o=self.Tno*self.a*np.arcsinh(self.slipv1/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
                #self.Tt2o=self.Tno*self.a*np.arcsinh(self.slipv2/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
            
            

            x=np.cos(self.rake)
            y=np.sin(self.rake)
            self.vec_Tra=np.array([np.cos(self.rake),np.sin(self.rake)]).transpose()
            
            # x=self.Tt1o/self.Tt
            # y=self.Tt2o/self.Tt
            # self.vec_Tra=np.array([x,y]).transpose()
            #print(self.vec_Tra.shape)

            self.fric=self.Tt/(self.Tno-self.P*1e-6)
            
            self.state=np.log(np.sinh(self.Tt/(self.Tno-self.P*1e-6)/self.a)*2.0*self.V0/self.slipv)*self.a

            #self.Tt=self.Tt*0.98
            self.Tt1o=self.Tt*x
            self.Tt2o=self.Tt*y
            
            #self.state1=np.log(np.sinh(self.Tt1o/self.Tno/self.a)*2.0*self.V0/self.slipv1)*self.a
            #self.state2=np.log(np.sinh(self.Tt2o/self.Tno/self.a)*2.0*self.V0/self.slipv2)*self.a
            #print(np.max(self.state),np.min(self.state))
            self.maxslipv0=np.max(self.slipv)
            self.rake0=np.copy(self.rake)
            self.calc_nucleaszie_cohesivezone()
            #self.randompatch()
        
 
    #read vtk file for initial condition if it start from previous results
    def read_vtk(self,fname):
        #K=450
        #mesh0 = pv.read("examples/case1/out/step%d.vtk"%K)
        mesh0 = pv.read(fname)
        self.rake=mesh0.cell_data['rake[Degree]']
        self.rake=self.rake/180.*np.pi
        self.Tt1o=mesh0.cell_data['Shear_1[MPa]']
        self.Tt2o=mesh0.cell_data['Shear_2[MPa]']
        self.Tt=mesh0.cell_data['Shear_[MPa]']
        self.Tno=mesh0.cell_data['Normal_[MPa]']

        self.slipv1=mesh0.cell_data['Slipv1[m/s]']
        self.slipv2=mesh0.cell_data['Slipv2[m/s]']
        self.slipv=mesh0.cell_data['Slipv[m/s]']
        
        
        self.slip=mesh0.cell_data['slip']
        self.slip1=mesh0.cell_data['slip1']
        self.slip2=mesh0.cell_data['slip2']

        # self.a=mesh0.cell_data['a']
        # self.b=mesh0.cell_data['b']
        # self.dc=mesh0.cell_data['dc']

        self.state=mesh0.cell_data['state']
        self.fric=mesh0.cell_data['fric']



    #read initial condition from outside files
    def read_parameter(self,fname):
        f=open(self.dirname+'/'+fname,'r')
        values=[]
        for line in f:
            tem=line.split()
            tem=np.array(tem).astype(float)
            values.append(tem)
        f.close()

        values=np.array(values)
        Ncell=self.eleVec.shape[0]
        self.rake=values[:Ncell,0]
        self.a=values[:Ncell,1]
        self.b=values[:Ncell,2]
        self.dc=values[:Ncell,3]
        self.f0=values[:Ncell,4]
        #self.Tt1o=values[:Ncell,5]*1e6
        #self.Tt2o=values[:Ncell,5]*0
        self.Tt=values[:Ncell,5]
        self.Tno=values[:Ncell,6]
        #self.slipv1=values[:Ncell,7]
        #self.slipv2=-values[:Ncell,7]*0.0
        self.slipv=values[:Ncell,7]
        #self.slipv=self.slipvC
        
        self.shear_loading=values[:Ncell,8]
        self.normal_loading=values[:Ncell,9]

        try:
            self.P=values[:Ncell,10]*1e6
        except:
            print('No Initial porepressure data!')
        #     return

        #slipv1=1e-9
        #self.Tt=self.Tno*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
        
        # x=self.Tt1o/self.Tt
        # y=self.Tt2o/self.Tt
        # self.rake=np.arctan2(y,x)
        #print(self.rake)
        #self.vec_Tra=np.array([x,y]).transpose()
        #print(self.vec_Tra.shape)
        #self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)

        
        x=np.cos(self.rake)
        y=np.sin(self.rake)
        #self.Tt1o=self.Tt*x
        #self.Tt2o=self.Tt*y+1.0
        self.slipv1=self.slipv*x+1e-16
        self.slipv2=self.slipv*y+1e-16
        
        self.vec_Tra=np.array([x,y]).transpose()
        self.Tt=(self.Tno-self.P*1e-6)*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
        #self.Tt=self.Tno*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
        #self.Tt1o=self.Tno*self.a*np.arcsinh(self.slipv1/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt2o=self.Tno*self.a*np.arcsinh(self.slipv2/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt2o=np.zeros(len(self.a))
        self.Tt1o=self.Tt*x
        self.Tt2o=self.Tt*y

        self.fric=self.Tt/(self.Tno-self.P*1e-6)
        self.state=np.log(np.sinh(self.Tt/(self.Tno-self.P*1e-6)/self.a)*2.0*self.V0/self.slipv)*self.a
        #self.state1=np.log(np.sinh(self.Tt1o/self.Tno/self.a)*2.0*self.V0/self.slipv1)*self.a
        #self.state2=np.log(np.sinh((self.Tt2o)/self.Tno/self.a)*2.0*self.V0/(self.slipv2))*self.a
        
        #print(self.state1,self.state2)
        T_globalarr=[]
        for i in range(len(self.rake)):
            T_local=np.zeros(3)
            T_local[0]=cos(self.rake[i])
            T_local[1]=sin(self.rake[i])
            Me=self.eleVec[i].reshape([3,3])
            T_global=np.dot(Me.transpose(),T_local)
            #print(self.Tt1o[i],self.Tt2o[i],T_global)
            T_globalarr.append(T_global)  
        self.T_globalarr=np.array(T_globalarr)

    #Partial derivative calculation
    def derivative(self,Tno,Tt1o,Tt2o,state):
        Tno=Tno*1e6
        Tt1o=Tt1o*1e6
        Tt2o=Tt2o*1e6
        P=self.P
        dPdt=self.dPdt0


        def safe_exp(x, max_value=700):  # 限制指数的最大值
            return np.exp(np.clip(x, -max_value, max_value))

        def safe_cosh(x, max_value=700):  # 使用指数形式的cosh，避免溢出
            x = np.clip(x, -max_value, max_value)
            return (np.exp(x) + np.exp(-x)) / 2

        def safe_sinh(x, max_value=700):  # 使用指数形式的sinh，避免溢出
            x = np.clip(x, -max_value, max_value)
            return (np.exp(x) - np.exp(-x)) / 2

        # 参数与公式
        V0 = self.V0
        a = self.a
        b = self.b
        dc = self.dc
        f0 = self.f0

        #theta=dc/V0*np.exp((state-f0)/b)
        #dthetadt=1.0-self.slipv*theta/dc
        #dthetadt=-theta*self.slipv/dc*np.log(theta*self.slipv/dc)
        # slipv1 = self.slipv1
        # slipv2 = self.slipv2
        # slipv_gpu=np.sqrt(slipv1*slipv1+slipv2*slipv2)

        #print('Tt1o / (a * Tno):',np.max(Tt1o / (a * Tno)))
        # 计算公式
        dV1dtau = 2 * V0 / (a * (Tno-P)) * safe_exp(-state / a) * safe_cosh(Tt1o / (a * (Tno-P)))
        dV2dtau = 2 * V0 / (a * (Tno-P)) * safe_exp(-state / a) * safe_cosh(Tt2o / (a * (Tno-P)))
        dV1dsigma = -2 * V0 * Tt1o / (a * (Tno-P)**2) * safe_exp(-state / a) * safe_cosh(Tt1o / (a * (Tno-P)))
        dV2dsigma = -2 * V0 * Tt2o / (a * (Tno-P)**2) * safe_exp(-state / a) * safe_cosh(Tt2o / (a * (Tno-P)))
        dV1dstate = -2 * V0 / a * safe_exp(-state / a) * safe_sinh(Tt1o / (a * (Tno-P)))
        dV2dstate = -2 * V0 / a * safe_exp(-state / a) * safe_sinh(Tt2o / (a * (Tno-P)))
        dstatedt = b / dc * (V0 * safe_exp((f0 - state) / b) - self.slipv)
        
        #print(np.max(-state/a+Tt1o / (a * Tno)),np.max(-state/a-Tt1o / (a * Tno)))
        #P=np.ones(len(Tno))*2000
        #print('P',np.max(np.abs(P)))

        #P=0
        # dV1dtau = 2 * V0 / (a * (Tno-P)) *0.5*(np.exp(-state/a+Tt1o / (a * (Tno-P)))+np.exp(-state/a-Tt1o / (a * (Tno-P))))
        # dV2dtau = 2 * V0 / (a * (Tno-P)) *0.5*(np.exp(-state/a+Tt2o / (a * (Tno-P)))+np.exp(-state/a-Tt2o / (a * (Tno-P))))
        # dV1dsigma = -2 * V0 * Tt1o / (a * (Tno-P)**2) * 0.5*(np.exp(-state/a+Tt1o / (a * (Tno-P)))+np.exp(-state/a-Tt1o / (a * (Tno-P))))
        # dV2dsigma = -2 * V0 * Tt2o / (a * (Tno-P)**2) * 0.5*(np.exp(-state/a+Tt2o / (a * (Tno-P)))+np.exp(-state/a-Tt2o / (a * (Tno-P))))
        # dV1dstate = -2 * V0 / a * 0.5*(np.exp(-state/a+Tt1o / (a * (Tno-P)))-np.exp(-state/a-Tt1o / (a * (Tno-P))))
        # dV2dstate = -2 * V0 / a * 0.5*(np.exp(-state/a+Tt2o / (a * (Tno-P)))-np.exp(-state/a-Tt2o / (a * (Tno-P))))
        # dstatedt = b / dc * (V0 * safe_exp((f0 - state) / b) - self.slipv)
        
        
        
        #dstatedt=b*self.slipv/dc*np.exp(2*(f0-state)/b)*(np.log(self.slipv/V0)+(state-f0)/b)
        
        #dPdt=(self.P0*1e6-P)/self.tf+self.DilatancyC/self.EPermeability*dthetadt/theta
        #dPdt=(self.P0*1e6-P)/self.tf+self.DilatancyC/dc/V0*dthetadt/theta
        #dPdt=self.DilatancyC/dc/V0*dthetadt/theta
        #dPdt=(self.P0*1e6-P)/self.tf+self.DilatancyC*1e-2*dthetadt/theta
        #dPdt=10
        #dV1dP=2 * V0 * Tt1o / (a * (Tno-P)**2) * 0.5*(np.exp(-state/a+Tt1o / (a * (Tno-P)))+np.exp(-state/a-Tt1o / (a * (Tno-P))))
        #dV2dP=2 * V0 * Tt2o / (a * (Tno-P)**2) * 0.5*(np.exp(-state/a+Tt2o / (a * (Tno-P)))+np.exp(-state/a-Tt2o / (a * (Tno-P))))
        #print('dV1dP:',np.max(dV1dP),np.min(dV1dP))
        #print('dV1dsigma:',np.max(dV1dsigma),np.min(dV1dsigma))  
        
        dtau1dt=(-self.AdotV1+self.shear_loading-self.mu/(2.0*self.Cs)*(dV1dsigma*(self.dsigmadt-dPdt)+dV1dstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dV1dtau)
        dtau2dt=(-self.AdotV2+self.shear_loading-self.mu/(2.0*self.Cs)*(dV2dsigma*(self.dsigmadt-dPdt)+dV2dstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dV2dtau)

        return dstatedt,self.dsigmadt*1e-6,dtau1dt*1e-6,dtau2dt*1e-6
    

    def Calc_Pmatrix(self):
        K=len(self.zp)
        z_k = self.zp
        c_hyd = self.Chyd 
        dz = np.diff(self.zp)
        #for j in range(len(self.eleVec)):
        for i in range(K-1):
            if(i==0):
                self.Pmatrix[0,0]=c_hyd*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
                self.Pmatrix[0,1]=-c_hyd*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
            else:
                self.Pmatrix[i,i]=c_hyd*(np.exp(-(2*z_k[i]-dz[i]/2))+np.exp(-(2*z_k[i]+dz[i]/2)))/(dz[i]*dz[i])
                self.Pmatrix[i,i+1]=-c_hyd*np.exp(-(2*z_k[i]+dz[i]/2))/(dz[i]*dz[i])
                self.Pmatrix[i,i-1]=-c_hyd*np.exp(-(2*z_k[i]-dz[i]/2))/(dz[i]*dz[i])

    def Calc_P_implicit(self,dt):
        e = self.DilatancyC  
        h = self.hs  
        beta = self.EPermeability    
        c_hyd = self.Chyd  
        dc=self.dc
        z_k = self.zp

        dz = np.diff(self.zp)
        #delta=self.slip
        K=len(self.zp)
        M=self.Pmatrix[:-1,:-1]*dt+np.eye(K-1)
        lu, piv = lu_factor(M)
        theta=self.dc/self.V0*np.exp((self.state-self.f0)/self.b)
        dthetadt=1.0-self.slipv*theta/dc
        #g=e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*self.state/dc)*np.exp(-delta/dc)
        #dstatedt = self.b / dc * (self.V0 * np.exp((self.f0 - self.state) / self.b) - self.slipv)
        g=-e*h/(2.0*beta*c_hyd*theta)*dthetadt
        #print('gmax:',np.max(self.slipv),'   gmin:',np.min(self.slipv))
        #B1=[]
        for i in range(len(self.eleVec)):
            b=np.copy(self.Parr[i,:-1])
            b[0]=b[0]-2.0*c_hyd*np.exp(-(z_k[0]-dz[0]/2))*g[i]*dt/dz[0]
            #B1.append(b[0])
            b[-1]=b[-1]+dt/(dz[-1]*dz[-1])*c_hyd*np.exp(-(z_k[-2]-dz[-1]/2))*self.P0
            x = lu_solve((lu, piv), b)
            self.Parr[i,:-1]=np.copy(x)
            #self.dPdt0[i]=(x[0]*1e-6-self.P[i])/dt
            term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (self.Parr[i,0] - self.Parr[i,1]+2*dz[0]*g[i]*exp(z_k[0]))
            term2 = np.exp(-(z_k[0] + dz[0]/2)) * (self.Parr[i,1] - self.Parr[i,0])
            self.dPdt0[i]=c_hyd * np.exp(-z_k[0]) * (term1 + term2) / dz[0]**2

            
            #self.dPdt0[i]=0
            self.P[i]=self.Parr[i,0]

    def Calc_P_implicit_mpi(self,dt,P,dPdt0,Parr):
        e = self.DilatancyC  
        h = self.hs  
        beta = self.EPermeability    
        c_hyd = self.Chyd  
        dc=self.dc
        # dc=self.dc[self.local_index]
        # state=self.state[self.local_index]
        # f0=self.f0[self.local_index]
        # b=self.b[self.local_index]
        # slipv=self.slipv[self.local_index]
        z_k = self.zp


        dz = np.diff(self.zp)
        #delta=self.slip
        K=len(self.zp)
        M=self.Pmatrix[:-1,:-1]*dt+np.eye(K-1)
        lu, piv = lu_factor(M)
        theta=self.dc/self.V0*np.exp((self.state-self.f0)/self.b)
        dthetadt=1.0-self.slipv*theta/dc
        #g=e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*self.state/dc)*np.exp(-delta/dc)
        #dstatedt = self.b / dc * (self.V0 * np.exp((self.f0 - self.state) / self.b) - self.slipv)
        g=-e*h/(2.0*beta*c_hyd*theta)*dthetadt
        #print('gmax:',np.max(g[self.local_index]),'   gmin:',np.min(g[self.local_index]))
        #B1=[]
        #print()
        index0=np.arange(0,len(self.eleVec),1)
        index_ = np.setdiff1d(index0, self.local_index)
        #print('self.local_index',self.local_index)
        Parr[index_]=0
        P[index_]=0
        dPdt0[index_]=0
        for i in range(len(self.local_index)):
            k=self.local_index[i]
            bv=np.copy(Parr[k,:-1])
            bv[0]=bv[0]-2.0*c_hyd*np.exp(-(z_k[0]-dz[0]/2))*g[k]*dt/dz[0]
            #B1.append(b[0])
            bv[-1]=bv[-1]+dt/(dz[-1]*dz[-1])*c_hyd*np.exp(-(z_k[-2]-dz[-1]/2))*self.P0
            x = lu_solve((lu, piv), bv)
            Parr[k,:-1]=np.copy(x)
            #self.dPdt0[i]=(x[0]*1e-6-self.P[i])/dt
            term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (Parr[k,0] - Parr[k,1]+2*dz[0]*g[k]*exp(z_k[0]))
            term2 = np.exp(-(z_k[0] + dz[0]/2)) * (Parr[k,1] - Parr[k,0])
            dPdt0[k]=c_hyd * np.exp(-z_k[0]) * (term1 + term2) / dz[0]**2

            #self.dPdt0[i]=0
            P[k]=Parr[k,0]
        #print('rank ',rank,np.max(P[self.local_index]),np.min(P[self.local_index]))
        return P,dPdt0,Parr


    def Calc_dP_DF(self,dt):
        e = self.DilatancyC  
        h = self.hs  
        beta = self.EPermeability    
        c_hyd = self.Chyd   
        #p_inf = self.P0  
        dc=self.dc
        delta=self.slip
        theta=self.dc/self.V0*np.exp((self.state-self.f0)/self.b)
        #g=-e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*theta/dc)*np.exp(-delta/dc)
        g=-e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*self.state/dc)
        K=len(self.zp)
        z_k = self.zp
        exp_neg_z_k = np.exp(-z_k)
        dp_dt=np.zeros(K)
        dz = np.diff(self.zp)
        for i in range(len(self.eleVec)):  # 模拟 1000 步
            # for k in range(0, K-1):
            #     z_k = self.zp[k]
            #     # 避免溢出，检查指数
            #     exp_neg_z_k = np.exp(-z_k)
            #     if(k>0):
            #         term1 = -np.exp(-(z_k - dz[k]/2)) * (self.Parr[i,k] - self.Parr[i,k-1])
            #         term2 = np.exp(-(z_k + dz[k]/2)) * (self.Parr[i,k+1] - self.Parr[k])
            #         dp_dt[k] = c_hyd * exp_neg_z_k * (term1 + term2) / dz[k]**2
            #     else:
            #         term1 = -np.exp(-(z_k - dz[k]/2)) * (self.Parr[i,k] - self.Parr[i,k+1]+2*dz[k]*g[i]*exp(z_k))
            #         term2 = np.exp(-(z_k + dz[k]/2)) * (self.Parr[i,k+1] - self.Parr[i,k])
            #         dp_dt[k]=c_hyd * exp_neg_z_k * (term1 + term2) / dz[k]**2
            term1 = -np.exp(-(z_k[1:] - dz/2)) * (self.Parr[i,1:] - self.Parr[i,:-1])
            term2 = np.exp(-(z_k[1:] + dz/2)) * (self.Parr[i,1:] - self.Parr[i,:-1])
            dp_dt[1:] = c_hyd * exp_neg_z_k[1:] * (term1 + term2) / dz**2

            term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (self.Parr[i,0] - self.Parr[i,1]+2*dz[0]*g[i]*exp(z_k[0]))
            term2 = np.exp(-(z_k[0] + dz[0]/2)) * (self.Parr[i,1] - self.Parr[i,0])
            dp_dt[0]=c_hyd * exp_neg_z_k[0] * (term1 + term2) / dz[0]**2

            self.Parr[i]=self.Parr[i]+dp_dt*dt
            self.dPdt0[i]=dp_dt[0]


    
    #forward modelling
    def simu_forward_mpi(self,dttry):
        
        slipv1=comm.bcast(self.slipv1, root=0)
        slipv2=comm.bcast(self.slipv2, root=0)
        
        if(self.Ifdila==True):
            Pre=comm.bcast(self.P, root=0)
            Parr=comm.bcast(self.Parr, root=0)
            dPdt0=comm.bcast(self.dPdt0, root=0)
        # index0=np.where(self.xg[:,2]<-8000)[0]
        # slipv1[index0]=slipv1[index0]-self.Vpl_con*np.cos(self.rake0[index0])
        # slipv2[index0]=slipv2[index0]-self.Vpl_con*np.sin(self.rake0[index0])


        slipv1=slipv1-self.slipvC*np.cos(self.rake0)
        slipv2=slipv2-self.slipvC*np.sin(self.rake0)
        #Calculating Kv first
        #comm.Barrier()
        if(self.fix_Tn==True):
            dsigmadt=self.normal_loading
        else:
            #dsigmadt=np.dot(self.Bs,slipv1)+np.dot(self.Bd,slipv2)+self.normal_loading
            dsigmadt=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'Bs')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'Bd')+self.normal_loading

        #dsigmadt[self.index_normal]=-dsigmadt[self.index_normal]
        AdotV1=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'A1s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'A1d')
        AdotV2=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'A2s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'A2d')

        #Combine results from all ranks
        self.dsigmadt=comm.reduce(dsigmadt, op=MPI.SUM, root=0)
        self.AdotV1=comm.reduce(AdotV1, op=MPI.SUM, root=0)
        self.AdotV2=comm.reduce(AdotV2, op=MPI.SUM, root=0)
        #comm.Barrier()
        #if(rank==0):
        #    print(self.AdotV2[100:120])
        nrjct=0
        h=dttry
        running=True
        dtnext=None
        if(rank==0):
            while running:
                #RungeKutte iteration and update tau,Tno, Tt1,Tt2
                Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk,condition1,condition2,hnew1,hnew2=self.RungeKutte_solve_Dormand_Prince(h)
                #print('P_yhk:',np.max(P_yhk),np.min(P_yhk))   
                #Tno_yhk,Tt_yhk,state_yhk,condition1,condition2,hnew1,hnew2=self.RungeKutta_solve6(h)
                #print('condition1,condition2,hnew1,hnew2:',condition1,condition2,hnew1,hnew2)
                
                #judge the accrancy
                if(max(condition1,condition2)<1.0 and not (np.isnan(condition1) or np.isnan(condition2))):
                    #print(type(hnew1),type(condition1))
                    dtnext=min(hnew1,hnew2)
                    dtnext=min(1.5*h,dtnext)
                    self.Relerrormax1_last=self.Relerrormax1
                    self.Relerrormax2_last=self.Relerrormax2
                    #print('dtnext:',dtnext)
                    break
                    
                    
                else:
                    nrjct=nrjct+1
                    dtnext=min(hnew1,hnew2)
                    h=max(0.5*h,dtnext)
                    #h=0.5*h
                    #print('nrjct:',nrjct,'  condition1,',condition1,' condition2:',condition2,'  dt:',h)

                    if(h<1.e-15 or nrjct>20):
                        print('error: dt is too small')
                        sys.exit()

            

            self.time=self.time+h
            #print(type(dtnext))
            #Tno_yhk[Tno_yhk<1.0]=1.0
            #Tt_yhk[Tt_yhk<0]=0
            self.Tno=Tno_yhk
            self.Tt1o=Tt1o_yhk
            self.Tt2o=Tt2o_yhk
            #self.P=P_yhk
            self.state=state_yhk
            #self.state2_gpu=state2_yhk
            
            #update slip rate and rake
            self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)

            #print('self.Tt1o',np.mean(self.Tt1o),np.mean(self.Tt2o))
            self.slipv1=(2.0*self.V0)*np.exp(-self.state/self.a)*np.sinh(self.Tt1o/self.Tno/self.a)
            self.slipv2=(2.0*self.V0)*np.exp(-self.state/self.a)*np.sinh(self.Tt2o/self.Tno/self.a)
            self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)
            self.rake=np.arctan2(self.Tt2o,self.Tt1o)
            
            indexmin=np.where(self.slipv<1e-30)[0]
            if(len(indexmin)>0):
                self.slipv[indexmin]=1e-30
            #self.slipv1_gpu=self.slipv_gpu*cp.cos(self.rake_gpu)
            #self.slipv2_gpu=self.slipv_gpu*cp.sin(self.rake_gpu)
            self.maxslipv0=np.max(self.slipv)
            
            #print('maxTt:',cp.max(Tt_yhk),'maxstate:',cp.max(state_yhk))
            #update slip
            self.slip1=self.slip1+self.slipv1*h
            self.slip2=self.slip2+self.slipv2*h
            self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
            
            #update frictional coeff
            self.fric=self.Tt/(self.Tno-self.P*1e-6)
        # else:
        #     h,dtnext=None,None
        
        comm.Barrier()
        h = comm.bcast(h, root=0)
        #if(self.step>0):
        dtnext = comm.bcast(dtnext, root=0)
        
        #update Pore pressure
        #self.Calc_dP_DF(h)
        #self.Calc_P_implicit(h)
        if(self.Ifdila==True):
            # if(rank==0):
            #     self.Calc_P_implicit(h)
            self.slipv=comm.bcast(self.slipv, root=0)
            self.state=comm.bcast(self.state, root=0)
            Pre,dPdt0,Parr=self.Calc_P_implicit_mpi(h,Pre,dPdt0,Parr)
            self.dPdt0=comm.reduce(dPdt0, op=MPI.SUM, root=0)
            self.P=comm.reduce(Pre, op=MPI.SUM, root=0)
            self.Parr=comm.reduce(Parr, op=MPI.SUM, root=0)
        return h,dtnext

    #forward modelling
    def simu_forward(self,dttry):
        slipv1=self.slipv1-self.slipvC*np.cos(self.rake0)
        slipv2=self.slipv2-self.slipvC*np.sin(self.rake0)
        #Calculating Kv first
        #comm.Barrier()
        if(self.fix_Tn==True):
            self.dsigmadt=self.normal_loading
        else:
            #dsigmadt=np.dot(self.Bs,slipv1)+np.dot(self.Bd,slipv2)+self.normal_loading
            self.dsigmadt=self.tree_block.blocks_process_MVM(slipv1,self.blocks_to_process,'Bs')+\
                self.tree_block.blocks_process_MVM(slipv2,self.blocks_to_process,'Bd')+self.normal_loading

        #dsigmadt[self.index_normal]=-dsigmadt[self.index_normal]
        self.AdotV1=self.tree_block.blocks_process_MVM(slipv1,self.blocks_to_process,'A1s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.blocks_to_process,'A1d')
        self.AdotV2=self.tree_block.blocks_process_MVM(slipv1,self.blocks_to_process,'A2s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.blocks_to_process,'A2d')

        

        nrjct=0
        h=dttry
        running=True
        dtnext=None

        while running:
            #RungeKutte iteration and update tau,Tno, Tt1,Tt2
            Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk,condition1,condition2,hnew1,hnew2=self.RungeKutte_solve_Dormand_Prince(h)
            #print('P_yhk:',np.max(P_yhk),np.min(P_yhk))   
            #Tno_yhk,Tt_yhk,state_yhk,condition1,condition2,hnew1,hnew2=self.RungeKutta_solve6(h)
            #print('condition1,condition2,hnew1,hnew2:',condition1,condition2,hnew1,hnew2)
            
            #judge the accrancy
            if(max(condition1,condition2)<1.0 and not (np.isnan(condition1) or np.isnan(condition2))):
                #print(type(hnew1),type(condition1))
                dtnext=min(hnew1,hnew2)
                dtnext=min(1.5*h,dtnext)
                self.Relerrormax1_last=self.Relerrormax1
                self.Relerrormax2_last=self.Relerrormax2
                #print('dtnext:',dtnext)
                break
                
                
            else:
                nrjct=nrjct+1
                dtnext=min(hnew1,hnew2)
                h=max(0.5*h,dtnext)
                #h=0.5*h
                #print('nrjct:',nrjct,'  condition1,',condition1,' condition2:',condition2,'  dt:',h)

                if(h<1.e-15 or nrjct>20):
                    print('error: dt is too small')
                    sys.exit()

        

        self.time=self.time+h
        #print(type(dtnext))
        #Tno_yhk[Tno_yhk<1.0]=1.0
        #Tt_yhk[Tt_yhk<0]=0
        self.Tno=Tno_yhk
        self.Tt1o=Tt1o_yhk
        self.Tt2o=Tt2o_yhk
        #self.P=P_yhk
        self.state=state_yhk
        #self.state2_gpu=state2_yhk
        
        #update slip rate and rake
        self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)

        #print('self.Tt1o',np.mean(self.Tt1o),np.mean(self.Tt2o))
        self.slipv1=(2.0*self.V0)*np.exp(-self.state/self.a)*np.sinh(self.Tt1o/self.Tno/self.a)
        self.slipv2=(2.0*self.V0)*np.exp(-self.state/self.a)*np.sinh(self.Tt2o/self.Tno/self.a)
        self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)
        self.rake=np.arctan2(self.Tt2o,self.Tt1o)
        
        indexmin=np.where(self.slipv<1e-30)[0]
        if(len(indexmin)>0):
            self.slipv[indexmin]=1e-30
        #self.slipv1_gpu=self.slipv_gpu*cp.cos(self.rake_gpu)
        #self.slipv2_gpu=self.slipv_gpu*cp.sin(self.rake_gpu)
        self.maxslipv0=np.max(self.slipv)
        
        #print('maxTt:',cp.max(Tt_yhk),'maxstate:',cp.max(state_yhk))
        #update slip
        self.slip1=self.slip1+self.slipv1*h
        self.slip2=self.slip2+self.slipv2*h
        self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
        
        #update frictional coeff
        self.fric=self.Tt/(self.Tno-self.P*1e-6)

        
        #update Pore pressure
        #self.Calc_dP_DF(h)
        #self.Calc_P_implicit(h)
        if(self.Ifdila==True):
            self.Calc_P_implicit(h)

        return h,dtnext


    #RungeKutte iteration
    def RungeKutte_solve_Dormand_Prince(self,h):
        B21=.2
        B31=3./40
        B32=9./40.

        B41=44./45.
        B42=-56./15
        B43=32./9

        B51=19372./6561.
        B52=-25360/2187.
        B53=64448./6561.
        B54=-212./729.

        B61=9017./3168.
        B62=-355./33.
        B63=-46732./5247.
        B64=49./176.
        B65=-5103./18656.

        B71=35./384.
        B73=500./1113.
        B74=125./192.
        B75=-2187./6784.
        B76=11./84.

        B81=5179./57600.
        B83=7571./16695.
        B84=393./640.
        B85=-92097./339200.
        B86=187./2100.
        B87=1./40.

        Tno=self.Tno
        Tt1o=self.Tt1o
        Tt2o=self.Tt2o
        state=self.state
        #P=self.P

        dstatedt1,dsigmadt1,dtau1dt1,dtau2dt1=self.derivative(Tno,Tt1o,Tt2o,state)
        
        
        #state2=self.state2_gpu
        Tno_yhk=Tno+h*B21*dsigmadt1
        Tt1o_yhk=Tt1o+h*B21*dtau1dt1
        Tt2o_yhk=Tt2o+h*B21*dtau2dt1
        #Tt2o_yhk=Tt2o+h*B21*dtau2dt1
        state_yhk=state+h*B21*dstatedt1
        #P_yhk=P+h*B21*dPdt1
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt2,dsigmadt2,dtau1dt2,dtau2dt2=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B31*dsigmadt1+B32*dsigmadt2)
        Tt1o_yhk=Tt1o+h*(B31*dtau1dt1+B32*dtau1dt2)
        Tt2o_yhk=Tt2o+h*(B31*dtau2dt1+B32*dtau2dt2)
        #Tt2o_yhk=Tt2o+h*(B31*dtau2dt1+B32*dtau2dt2)
        state_yhk=state+h*(B31*dstatedt1+B32*dstatedt2)
        #P_yhk=P+h*(B31*dPdt1+B32*dPdt2)
        #state2_yhk=state2+h*(B31*dstate2dt1+B32*dstate2dt2)
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt3,dsigmadt3,dtau1dt3,dtau2dt3=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B41*dsigmadt1+B42*dsigmadt2+B43*dsigmadt3)
        Tt1o_yhk=Tt1o+h*(B41*dtau1dt1+B42*dtau1dt2+B43*dtau1dt3)
        Tt2o_yhk=Tt2o+h*(B41*dtau2dt1+B42*dtau2dt2+B43*dtau2dt3)
        state_yhk=state+h*(B41*dstatedt1+B42*dstatedt2+B43*dstatedt3)
        #P_yhk=P+h*(B41*dPdt1+B42*dPdt2+B43*dPdt3)
        #state2_yhk=state2+h*(B41*dstate2dt1+B42*dstate2dt2+B43*dstate2dt3)
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt4,dsigmadt4,dtau1dt4,dtau2dt4=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B51*dsigmadt1+B52*dsigmadt2+B53*dsigmadt3+B54*dsigmadt4)
        Tt1o_yhk=Tt1o+h*(B51*dtau1dt1+B52*dtau1dt2+B53*dtau1dt3+B54*dtau1dt4)
        Tt2o_yhk=Tt2o+h*(B51*dtau2dt1+B52*dtau2dt2+B53*dtau2dt3+B54*dtau2dt4)
        state_yhk=state+h*(B51*dstatedt1+B52*dstatedt2+B53*dstatedt3+B54*dstatedt4)
        #P_yhk=P+h*(B51*dPdt1+B52*dPdt2+B53*dPdt3+B54*dPdt4)
        #state2_yhk=state2+h*(B51*dstate2dt1+B52*dstate2dt2+B53*dstate2dt3+B54*dstate2dt4)

        dstatedt5,dsigmadt5,dtau1dt5,dtau2dt5=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B61*dsigmadt1+B62*dsigmadt2+B63*dsigmadt3+B64*dsigmadt4+B65*dsigmadt5)
        Tt1o_yhk=Tt1o+h*(B61*dtau1dt1+B62*dtau1dt2+B63*dtau1dt3+B64*dtau1dt4+B65*dtau1dt5)
        Tt2o_yhk=Tt2o+h*(B61*dtau2dt1+B62*dtau2dt2+B63*dtau2dt3+B64*dtau2dt4+B65*dtau2dt5)
        state_yhk=state+h*(B61*dstatedt1+B62*dstatedt2+B63*dstatedt3+B64*dstatedt4+B65*dstatedt5)
        #P_yhk=P+h*(B61*dPdt1+B62*dPdt2+B63*dPdt3+B64*dPdt4+B65*dPdt5)
        #state2_yhk=state2+h*(B61*dstate2dt1+B62*dstate2dt2+B63*dstate2dt3+B64*dstate2dt4+B65*dstate2dt5)
        

        dstatedt6,dsigmadt6,dtau1dt6,dtau2dt6=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B71*dsigmadt1+B73*dsigmadt3+B74*dsigmadt4+B75*dsigmadt5+B76*dsigmadt6)
        Tt1o_yhk=Tt1o+h*(B71*dtau1dt1+B73*dtau1dt3+B74*dtau1dt4+B75*dtau1dt5+B76*dtau1dt6)
        Tt2o_yhk=Tt2o+h*(B71*dtau2dt1+B73*dtau2dt3+B74*dtau2dt4+B75*dtau2dt5+B76*dtau2dt6)
        state_yhk=state+h*(B71*dstatedt1+B73*dstatedt3+B74*dstatedt4+B75*dstatedt5+B76*dstatedt6)
        #P_yhk=P+h*(B71*dPdt1+B73*dPdt3+B74*dPdt4+B75*dPdt5+B76*dPdt6)
        #print('dstatedt6',np.max(dstatedt6),np.min(dstatedt6))
        #state2_yhk=state2+h*(B71*dstate2dt1+B73*dstate2dt3+B74*dstate2dt4+B75*dstate2dt5+B76*dstate2dt6)

        dstatedt7,dsigmadt7,dtau1dt7,dtau2dt7=self.derivative(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk8=Tno+h*(B81*dsigmadt1+B83*dsigmadt3+B84*dsigmadt4+B85*dsigmadt5+B86*dsigmadt6+B87*dsigmadt7)
        Tt1o_yhk8=Tt1o+h*(B81*dtau1dt1+B83*dtau1dt3+B84*dtau1dt4+B85*dtau1dt5+B86*dtau1dt6+B87*dtau1dt7)
        Tt2o_yhk8=Tt2o+h*(B81*dtau2dt1+B83*dtau2dt3+B84*dtau2dt4+B85*dtau2dt5+B86*dtau2dt6+B87*dtau2dt7)
        state_yhk8=state+h*(B81*dstatedt1+B83*dstatedt3+B84*dstatedt4+B85*dstatedt5+B86*dstatedt6+B87*dstatedt7)
        #P_yhk8=P+h*(B81*dPdt1+B83*dPdt3+B84*dPdt4+B85*dPdt5+B86*dPdt6+B87*dPdt7)
        
        #state2_yhk8=state2+h*(B81*dstate2dt1+B83*dstate2dt3+B84*dstate2dt4+B85*dstate2dt5+B86*dstate2dt6+B87*dstate2dt7)

        #state1_yhk_err=cp.abs(state1_yhk8-state1_yhk)
        state_yhk_err=np.abs(state_yhk8-state_yhk)
        Tno_yhk_err=np.abs(Tno_yhk8-Tno_yhk)
        #P_yhk_err=np.abs(P_yhk8-P_yhk)
        Tt1o_yhk_err=np.abs(Tt1o_yhk8-Tt1o_yhk)
        Tt2o_yhk_err=np.abs(Tt2o_yhk8-Tt2o_yhk)
        


        self.Relerrormax1=np.max(np.abs(state_yhk_err/state_yhk8))+1e-10

        
        Relerrormax2=np.max(np.abs(Tt1o_yhk_err/Tt1o_yhk8))
        Relerrormax2o=np.max(np.abs(Tt2o_yhk_err/Tt2o_yhk8))
        Relerrormax2_=np.max(np.abs(Tno_yhk_err/Tno_yhk8))
        #Relerrormax2_P=np.max(np.abs(P_yhk_err/P_yhk8))
        #print('error: ',np.max(Relerrormax2),np.max(Relerrormax2o),np.max(Relerrormax2_),np.max(Relerrormax2_P))

        #self.Relerrormax2=max(Relerrormax2,Relerrormax2o,Relerrormax2_,Relerrormax2_P)+1e-10
        self.Relerrormax2=max(Relerrormax2,Relerrormax2o,Relerrormax2_)+1e-10
        #self.Relerrormax2=cp.linalg.norm(Tt1o_yhk_err/Tt1o_yhk8)+1e-10

        #print('errormax1,errormax2,relaemax1,relaemax2:',errormax1,errormax2,self.Relerrormax1,self.Relerrormax2)

        if((self.maxslipv0)>1e-6):
            self.RelTol1=1e-4
            self.RelTol2=1e-4
        else:
            self.RelTol1=2e-6
            self.RelTol2=2e-6
        
        self.RelTol1=1e-4
        self.RelTol2=1e-4

        condition1=self.Relerrormax1/self.RelTol1
        condition2=self.Relerrormax2/self.RelTol2

        #print(self.Relerrormax1,self.Relerrormax2)

        hnew1=h*0.9*(self.RelTol1/self.Relerrormax1)**0.2
        hnew2=h*0.9*(self.RelTol2/self.Relerrormax2)**0.2


        return Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk,condition1,condition2,hnew1,hnew2

    
      
    
    

    #def ouputVTK(self,**kwargs):
    def ouputVTK(self,fname):
        Nnode=self.nodelst.shape[0]
        Nele=self.elelst.shape[0]
        f=open(fname,'w')
        f.write('# vtk DataFile Version 3.0\n')
        f.write('test\n')
        f.write('ASCII\n')
        f.write('DATASET  UNSTRUCTURED_GRID\n')
        f.write('POINTS '+str(Nnode)+' float\n')
        for i in range(Nnode):
            f.write('%f %f %f\n'%(self.nodelst[i][0],self.nodelst[i][1],self.nodelst[i][2]))
        f.write('CELLS '+str(Nele)+' '+str(Nele*4)+'\n')
        for i in range(Nele):
            f.write('3 %d %d %d\n'%(self.elelst[i][0]-1,self.elelst[i][1]-1,self.elelst[i][2]-1))
        f.write('CELL_TYPES '+str(Nele)+'\n')
        for i in range(Nele):
            f.write('5 ')
        f.write('\n')
        

        f.write('CELL_DATA %d ' %(Nele))
        f.write('SCALARS Normal_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tno)):
            f.write('%f '%(self.Tno[i]))
        f.write('\n')
        f.write('SCALARS Pore_pressure[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.P)):
            f.write('%f '%(self.P[i]*1e-6))
        f.write('\n')

        # f.write('SCALARS Shear1_[MPa] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.Tt1o)):
        #     f.write('%f '%(self.Tt1o[i]))
        # f.write('\n')

        # f.write('SCALARS Shear2_[MPa] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.Tt2o)):
        #     f.write('%f '%(self.Tt2o[i]))
        # f.write('\n')

        f.write('SCALARS Shear_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt[i]))
        f.write('\n')

        f.write('SCALARS Shear_1[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt1o[i]))
        f.write('\n')

        f.write('SCALARS Shear_2[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt2o[i]))
        f.write('\n')

        f.write('SCALARS rake[Degree] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.rake)):
            f.write('%f '%(self.rake[i]*180./np.pi))
        f.write('\n')


        # f.write('SCALARS Slipv1_[m/s] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.slipv1)):
        #     f.write('%f '%(self.slipv1[i]))
        # f.write('\n')

        # f.write('SCALARS Slipv2_[m/s] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.slipv2)):
        #     f.write('%f '%(self.slipv2[i]))
        # f.write('\n')


        f.write('SCALARS state float\nLOOKUP_TABLE default\n')
        for i in range(len(self.state)):
            f.write('%f '%(self.state[i]))
        f.write('\n')


        f.write('SCALARS Slipv[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv[i]))
        f.write('\n')

        f.write('SCALARS Slipv1[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv1[i]))
        f.write('\n')

        f.write('SCALARS Slipv2[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv2[i]))
        f.write('\n')

        f.write('SCALARS a float\nLOOKUP_TABLE default\n')
        for i in range(len(self.a)):
            f.write('%f '%(self.a[i]))
        f.write('\n')

        f.write('SCALARS b float\nLOOKUP_TABLE default\n')
        for i in range(len(self.b)):
            f.write('%f '%(self.b[i]))
        f.write('\n')

        f.write('SCALARS a-b float\nLOOKUP_TABLE default\n')
        for i in range(len(self.b)):
            f.write('%f '%(self.a[i]-self.b[i]))
        f.write('\n')

        f.write('SCALARS dc float\nLOOKUP_TABLE default\n')
        for i in range(len(self.dc)):
            f.write('%.10f '%(self.dc[i]))
        f.write('\n')

        f.write('SCALARS fric float\nLOOKUP_TABLE default\n')
        for i in range(len(self.fric)):
            f.write('%f '%(self.fric[i]))
        f.write('\n')


        f.write('SCALARS slip float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip[i]))
        f.write('\n')

        f.write('SCALARS slip1 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip1[i]))
        f.write('\n')

        f.write('SCALARS slip2 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip2[i]))
        f.write('\n')

        f.write('SCALARS slip_plate float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipvC)):
            f.write('%.15f '%(self.slipvC[i]))
        f.write('\n')


        # Val_arr=[]
        # for key, value in kwargs.items():
        #     Val_arr.append(value)
        # f.write('CELL_DATA '+str(Nele)+'\n')
        # f.write('SCALARS stress double '+str(len(kwargs))+'\n')
        # f.write('LOOKUP_TABLE default\n')
        # for i in range(Nele):
        #     for j in range(len(Val_arr)):
        #         f.write('%f ' %(Val_arr[j][i]))
        #     f.write('\n')
        # f.write('\n')
        f.close()

    
        

    def get_value(self,x,y,z):
        Radius=np.linalg.norm(self.xg-np.array([x,y,z]),axis=1)
        index1_source = np.argsort(Radius)[0]
        return index1_source



    def outputtxt(self,fname):
        directory='out_txt'
        if not os.path.exists(directory):
            os.mkdir(directory)

        xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
        zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])
        X1=np.linspace(xmin+self.maxsize,xmax-self.maxsize,500)
        Y1=np.linspace(zmin+self.maxsize,zmax-self.maxsize,300)
        #for i in range(self.xg):
        X_grid, Y_grid = np.meshgrid(X1, Y1)
        X=X_grid.flatten()
        Y=Y_grid.flatten()
        mesh1 = np.column_stack((X, Y))
        #print(self.xg[[0,2]].shape, self.slipv.shape)
        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        #plt.pcolor(slipv_mesh)
        #plt.show()
        f=open(directory+'/X_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%X_grid[i][j])
            f.write('\n')
        f.close()

        f=open(directory+'/Y_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%Y_grid[i][j])
            f.write('\n')
        f.close()


        f=open(directory+'/'+fname+'slipv'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()


        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv1, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv1'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv2, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv2'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.Tt, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'Traction'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        # plt.pcolor(slipv_mesh)
        # plt.show()





            
        
        










    #def get_coreD()

        #self.readdata(fname)
        #a=self.external_header_length
        #self.data = data
        # 在这里可以进行一些初始化操作
        
    