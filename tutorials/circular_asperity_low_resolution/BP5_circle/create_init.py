import numpy as np
import matplotlib.pyplot as plt
from math import *
import os
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 16})

pi=3.1415926



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
    
    return np.array(node)*1e3,np.array(ele)
    

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
        rr = sqrt(ev31*ev31+ev32*ev32+ev33*ev33)
        # unit vectors for local coordinates of elements
        ev31 /=rr
        ev32 /=rr
        ev33 /= rr

        if( abs(ev33) < 1 ):
            ev11 = ev32
            ev12 = -ev31
            ev13 = 0 
            rr = sqrt(ev11*ev11 + ev12*ev12) 
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



fnamegeo='examples/BP5_circle/aspirity_circle.msh'
jud_ele_order=False
nodelst,elelst=read_mshV2(fnamegeo)
eleVec,xg=get_eleVec(nodelst,elelst,jud_ele_order)



alst=np.zeros(eleVec.shape[0])
blst=np.zeros(eleVec.shape[0])


angle=0.0*np.pi/180.0
f0=0.6
slipv=1e-9
V0=1e-6
slipvC=1.1e-9
Dc=0.1
Tn=40
Tt=14.0/25*40

target_point=0,0,-25000

distances = np.sqrt(np.sum((xg - target_point) ** 2, axis=1))

Ds=18000
avs=0.04
bvs=0.03
avw=0.01
bvw=0.03
# index0=np.where(distances<Ds)[0]
# index1=np.where(distances>=Ds)[0]
# alst[index1]=0.04
# blst[index1]=0.03

# alst[index0]=0.01
# blst[index0]=0.03
# Wedge=5000
for i in range(len(alst)):
    #tem=min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge
    coords1=np.array([xg[i]])
    
    dis=distances[i]/Ds

    
    
    
    

    if(dis<0.7):
        #print('!!!!!',dis)
        alst[i]=avw
        blst[i]=bvw
        
    elif(dis<1.0):
        
        dis1=(dis-0.7)/0.3
        alst[i]=avs-(avs-avw)*(1.0-dis1)
        blst[i]=bvs

        
    
    else:
        alst[i]=avs
        blst[i]=bvs


print(len(xg))
f=open('Init_circle_asp.dat','w')
for i in range(len(xg)):
    f.write('0.0 %f %f %f %f %f %f 1.1e-9 0.0 0.0 0.0\n' %(alst[i],blst[i],Dc,f0,Tt,Tn))
f.close()

