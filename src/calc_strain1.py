import readmsh
import SH_greenfunction
import numpy as np
import ctypes
from math import *

lib = ctypes.CDLL('src/TDstressFS_C.so')
#lib = ctypes.CDLL('src/Dll1.dll')
# 定义接口参数类型
lib.TDstressFS_C.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # X
    ctypes.POINTER(ctypes.c_double),  # Y
    ctypes.POINTER(ctypes.c_double),  # Z
    ctypes.c_size_t,                  # n
    ctypes.POINTER(ctypes.c_double),  # P1
    ctypes.POINTER(ctypes.c_double),  # P2
    ctypes.POINTER(ctypes.c_double),  # P3
    ctypes.c_double, ctypes.c_double, ctypes.c_double,  # Ss, Ds, Ts
    ctypes.c_double, ctypes.c_double,                  # mu, lambda_
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_double),  # stress
    ctypes.POINTER(ctypes.c_double),  # strain
]

v=0.25
p=2670
vs=3464

mu=p*vs*vs
lamuda=2*mu*v/(1.0-2.0*v)

print(mu,lamuda)

fnamegeo='examples/Lab-model/lab.msh'
nodelst,elelst=readmsh.read_mshV2(fnamegeo)
nodelst=nodelst/1e3
jud_ele_order=False
eleVec,xg=readmsh.get_eleVec(nodelst,elelst,jud_ele_order)

def read_data(fname,headline):
    f=open(fname,'r')
    linenum=0
    Data=[]
    for line in f:
        linenum=linenum+1
        if(linenum>=headline):
            try:
                tem=np.array(line.split()).astype(float)
                Data.append(tem)
            except:
                continue
    Data=np.array(Data)
    return Data

fname='examples/Lab-model/state.txt'
headline=5
para=read_data(fname,headline)
Time_sec=np.cumsum(para[:,1])
time_interval=para[:,1]
#start_time=565.5
start_time=493
#end_time=568
end_time=645.3
#end_time=660
#end_time=720
index0=np.where((Time_sec>start_time) & (Time_sec<end_time))[0]
print(index0.shape)
DATA_slipv=[]
for i in range(50):
#for i in range(1000):
    K=i*200
    print(i)
    #data0=np.load('slip/slip_%d.npy'%K)
    #DATA_slip.append(data0)
    data0=np.load('examples/Lab-model/out_slipvTt/slipv_%d.npy'%K)
    #slipv_max=np.max(data0,axis=1)
    DATA_slipv.append(data0)
    
DATA_slipv=np.concatenate(DATA_slipv,axis=0)


Slip=np.dot(DATA_slipv[index0].transpose(),time_interval[index0])

print(np.max(Slip),np.min(Slip))


index1=np.arange(0,len(xg),3)
X,Y=xg[index1, 0], xg[index1, 1]

# x = np.linspace(np.min(xg[:,0]), np.max(xg[:,0]), 100)  # X 方向 100 个点
# y = np.linspace(np.min(xg[:,1]), np.max(xg[:,1]),  100)    # Y 方向 50 个点

# X, Y = np.meshgrid(x, y) 
Z=np.arange(-0.08,-0.02,0.0005)
#Z=[-0.05]
scale1 = np.random.normal(loc=0.0, scale=0.1, size=len(X))

X_grid=[]
Y_grid=[]
Z_grid=[]

f=open('examples/Lab-model/point.txt','w')
f.write('x,y,z\n')
for i in range(len(Z)):
    for j in range(len(X)):
        f.write('%f,%f,%f\n'%(X[j],Y[j],Z[i]))
        X_grid.append(X[j])
        Y_grid.append(Y[j])
        Z_grid.append(Z[i])
f.close()


X_grid=np.array(X_grid)
Y_grid=np.array(Y_grid)
Z_grid=np.array(Z_grid)
n=len(X_grid)


# strain_sum=np.load('strain_sum.npy')
# print(strain_sum)
# strain_sum=strain_sum.reshape([n,6])
# print(strain_sum.shape)

# f=open('examples/lab/point.txt','w')
# f.write('x,y,z,Ezz,Etheta\n')
# E_theta_lst=[]
# for i in range(len(X_grid)):
    
#     r=sqrt(X_grid[i]*X_grid[i]+Y_grid[i]*Y_grid[i])
#     sintheta=Y_grid[i]/r
#     costheta=X_grid[i]/r
#     E_theta=strain_sum[i,0]*sintheta*sintheta+strain_sum[i,1]*costheta*costheta-2.0*strain_sum[i,3]*sintheta*costheta
#     E_theta_lst.append(E_theta)
#     #print(strain_sum[i],E_theta)
#     f.write('%f,%f,%f,%f,%f\n'%(X_grid[i],Y_grid[i],Z_grid[i],strain_sum[i,2],E_theta))

# f.close()

def ouputVTK2d(nodelst,elelst,Exx,Eyy,Ezz,Et,fname):
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
    
    
    f.write('POINT_DATA %d\n' % (Nnode))     # 节点数量

    # 输出第一个标量字段
    f.write('SCALARS Exx float\nLOOKUP_TABLE default\n')
    for i in range(len(Exx)):
        f.write('%.10f ' % (Exx[i]))
    f.write('\n')

    f.write('SCALARS Eyy float\nLOOKUP_TABLE default\n')
    for i in range(len(Eyy)):
        f.write('%.10f ' % (Eyy[i]))
    f.write('\n')

    f.write('SCALARS Ezz float\nLOOKUP_TABLE default\n')
    for i in range(len(Ezz)):
        f.write('%.10f ' % (Ezz[i]))
    f.write('\n')

    # 输出第二个标量字段
    f.write('SCALARS Ethea float\nLOOKUP_TABLE default\n')
    for i in range(len(Et)):
        f.write('%.10f ' % (Et[i]))
    f.write('\n')


def ouputVTK(nodelst,elelst,Exx,Eyy,Ezz,Et,fname):
    Nnode=nodelst.shape[0]
    Nele=elelst.shape[0]
    f=open(fname,'w')
    f.write('# vtk DataFile Version 3.0\n')
    f.write('Tetrahedral mesh\n')
    f.write('ASCII\n')
    f.write('DATASET  UNSTRUCTURED_GRID\n')
    f.write('POINTS '+str(Nnode)+' float\n')
    for i in range(Nnode):
        f.write('%f %f %f\n'%(nodelst[i][0],nodelst[i][1],nodelst[i][2]))
    f.write('CELLS '+str(Nele)+' '+str(Nele*5)+'\n')
    for i in range(Nele):
        f.write('4 %d %d %d %d\n'%(elelst[i][0]-1,elelst[i][1]-1,elelst[i][2]-1,elelst[i][3]-1))
    
    
    f.write('CELL_TYPES '+str(Nele)+'\n')
    for i in range(Nele):
        f.write('10 ')
    f.write('\n')
    
    

    f.write('POINT_DATA %d\n' % (Nnode))     # 节点数量

    # 输出第一个标量字段
    f.write('SCALARS Exx float\nLOOKUP_TABLE default\n')
    for i in range(len(Exx)):
        f.write('%.10f ' % (Exx[i]))
    f.write('\n')

    f.write('SCALARS Eyy float\nLOOKUP_TABLE default\n')
    for i in range(len(Eyy)):
        f.write('%.10f ' % (Eyy[i]))
    f.write('\n')

    f.write('SCALARS Ezz float\nLOOKUP_TABLE default\n')
    for i in range(len(Ezz)):
        f.write('%.10f ' % (Ezz[i]))
    f.write('\n')

    
    # 输出第二个标量字段
    f.write('SCALARS Ethea float\nLOOKUP_TABLE default\n')
    for i in range(len(Et)):
        f.write('%.10f ' % (Et[i]))
    f.write('\n')
    

def is_float_convertible(x):
    try:
        float(x)
        return True
    except:
        return False

def readinp(fname):
    start_node=False
    start_ele=False
    nodelst=[]
    elelst=[]
    nodedict={}
    f=open(fname,'r')
    for line in f:
        if(start_node==True):
            vec_func = np.vectorize(is_float_convertible)
            result = vec_func(line.split(','))
            jud=np.any(result == False)
            if(jud==True):
                start_node=False
                print('node end',line,len(nodelst))
            else:
                tem=np.array(line.split(',')).astype(float)
                nodelst.append(tem[1:])
                nodedict[int(tem[0])]=tem[1:]
        if(start_ele==True):
            vec_func = np.vectorize(is_float_convertible)
            result = vec_func(line.split(','))
            jud=np.any(result == False)
            if(jud==True):
                start_ele=False
                print('ele end',line,len(elelst))
            else:
                elelst.append(np.array(line.split(','))[1:].astype(int))
        if(line=='*NODE\n' and len(nodelst)==0):
            start_node=True
            print('start node',len(nodelst))
        if(line[:8]=='*ELEMENT' and len(elelst)==0):
            start_ele=True
            print('start ele',len(elelst))
    return np.array(nodelst),np.array(elelst),nodedict

fname='examples/Lab-model/geo/cylinder.inp'
nodelstTET,elelstTET,nodedict=readinp(fname)

print('nodelstTET',nodelstTET.shape,elelstTET.shape)

nodelstTET=nodelstTET/100
nodelstTET[:,2]=nodelstTET[:,2]-0.1

t=1
nodedict1={}
for k, v in nodedict.items():
    nodedict1[k] = t
    t=t+1





xgTET=[]
for i in range(len(elelstTET)):
    xa=nodelstTET[elelstTET[i,0]-1]
    xb=nodelstTET[elelstTET[i,1]-1]
    xc=nodelstTET[elelstTET[i,2]-1]
    xd=nodelstTET[elelstTET[i,3]-1]
    xgTET.append([np.mean([xa[0],xb[0],xc[0],xd[0]]),np.mean([xa[1],xb[1],xc[1],xd[1]]),np.mean([xa[2],xb[2],xc[2],xd[2]])])
xgTET=np.array(xgTET)



# n=len(elelstTET)
# strain_sum=np.load('strain_sum.npy')
# strain_sum=strain_sum.reshape([n,6])
# print(strain_sum.shape,elelstTET.shape)
# Ezz=strain_sum[:,2]





# E_theta_lst=[]
# for i in range(len(xgTET)):
    
#     r=sqrt(xgTET[i,0]*xgTET[i,0]+xgTET[i,1]*xgTET[i,1])
#     sintheta=xgTET[i,1]/r
#     costheta=xgTET[i,0]/r
#     E_theta=strain_sum[i,0]*sintheta*sintheta+strain_sum[i,1]*costheta*costheta-2.0*strain_sum[i,3]*sintheta*costheta
#     E_theta_lst.append(E_theta)
#     #print(strain_sum[i],E_theta)
    
#ouputVTK(nodelstTET,elelstTET,xgTET[:,0],xgTET[:,2],fname='examples/lab/cylinder.vtk')







# n=len(X_grid)

# strain_sum= np.zeros(n * 6)
# stress_sum= np.zeros(n * 6)
# stress = np.zeros(n * 6)
# strain = np.zeros(n * 6)
# mu_,lambda_=30038120320.0,30038120320.0
# for i in range(len(elelst)):
# #for i in range(5):
#     print('calc stress ',i)
#     P1=np.copy(nodelst[elelst[i,0]-1])
#     P2=np.copy(nodelst[elelst[i,1]-1])
#     P3=np.copy(nodelst[elelst[i,2]-1])
    
#     Ss,Ds,Ts=0,Slip[i],0
#     judhalf=False
#     lib.TDstressFS_C(X_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 Y_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 Z_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 n,
#                 P1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 P2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 P3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 Ss,Ds,Ts,   # Ss, Ds, Ts
#                 mu_,lambda_,       # mu, lambda
#                 judhalf,
#                 stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#                 strain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#                 )
    
#     strain_sum=strain_sum+strain
#     stress_sum=stress_sum+stress
#     #print(strain_sum,stress_sum)
# #stress=stress.reshape([n,6])
# #stress=stress.transpose()
# np.save('strain_sum645',strain_sum)
# np.save('stress_sum645',stress_sum)


n=len(X_grid)
strain_sum=np.load('strain_sum645.npy')
strain_sum=strain_sum.reshape([n,6])
# print(strain_sum.shape,nodelstTET.shape)
# Ezz=strain_sum[:,2]

# ouputVTK(nodelstTET,elelstTET,strain_sum[:,2],strain_sum[:,1],fname='examples/lab/cylinder.vtk')


f=open('examples/Lab-model/point.txt','w')
f.write('x,y,z,Ezz,Etheta\n')
E_theta_lst=[]
for i in range(len(X_grid)):
    
    r=sqrt(X_grid[i]*X_grid[i]+Y_grid[i]*Y_grid[i])
    sintheta=Y_grid[i]/r
    costheta=X_grid[i]/r
    E_theta=strain_sum[i,0]*sintheta*sintheta+strain_sum[i,1]*costheta*costheta-2.0*strain_sum[i,3]*sintheta*costheta
    E_theta_lst.append(E_theta)
    #print(strain_sum[i],E_theta)
    f.write('%f,%f,%f,%f,%f\n'%(X_grid[i],Y_grid[i],Z_grid[i],strain_sum[i,2],E_theta))

f.close()


from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

Ezz = gaussian_filter(strain_sum[:,2], sigma=1.0)
E_theta = gaussian_filter(E_theta_lst, sigma=1.0)

points = np.vstack((X_grid, Y_grid, Z_grid)).T
grid_Exx = griddata(points, strain_sum[:,0], (nodelstTET[:,0], nodelstTET[:,1], nodelstTET[:,2]), method='nearest')
grid_Eyy = griddata(points, strain_sum[:,1], (nodelstTET[:,0], nodelstTET[:,1], nodelstTET[:,2]), method='nearest')

grid_Ezz = griddata(points, strain_sum[:,2], (nodelstTET[:,0], nodelstTET[:,1], nodelstTET[:,2]), method='nearest')
grid_E_theta = griddata(points, E_theta_lst, (nodelstTET[:,0], nodelstTET[:,1], nodelstTET[:,2]), method='nearest')

ouputVTK(nodelstTET,elelstTET,grid_Exx,grid_Eyy,grid_Ezz,grid_E_theta,fname='examples/Lab-model/cylinder.vtk')


fname='examples/Lab-model/geo/cylinder_face.inp'
nodelstTET2d,elelstTET2d,nodedict2d=readinp(fname)

print('nodelstTET',nodelstTET2d.shape,elelstTET2d.shape)

nodelstTET2d=nodelstTET2d/100
nodelstTET2d[:,2]=nodelstTET2d[:,2]-0.1

t=1
nodedict1={}
for k, v in nodedict2d.items():
    nodedict1[k] = t
    t=t+1

for i in range(len(elelstTET2d)):
    a=elelstTET2d[i][0]
    b=elelstTET2d[i][1]
    c=elelstTET2d[i][2]
    elelstTET2d[i][0]=nodedict1[a]
    elelstTET2d[i][1]=nodedict1[b]
    elelstTET2d[i][2]=nodedict1[c]
    #print(elelstTET2d[i],nodedict1[tem])
    



grid_Exx = griddata(points, strain_sum[:,0], (nodelstTET2d[:,0], nodelstTET2d[:,1], nodelstTET2d[:,2]), method='nearest')
grid_Eyy = griddata(points, strain_sum[:,1], (nodelstTET2d[:,0], nodelstTET2d[:,1], nodelstTET2d[:,2]), method='nearest')

grid_Ezz = griddata(points, strain_sum[:,2], (nodelstTET2d[:,0], nodelstTET2d[:,1], nodelstTET2d[:,2]), method='nearest')
grid_E_theta = griddata(points, E_theta_lst, (nodelstTET2d[:,0], nodelstTET2d[:,1], nodelstTET2d[:,2]), method='nearest')
ouputVTK2d(nodelstTET2d,elelstTET2d,grid_Exx,grid_Eyy,grid_Ezz,grid_E_theta,fname='examples/Lab-model/cylinder2d_720.vtk')
