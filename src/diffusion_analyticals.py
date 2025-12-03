import readmsh
import numpy as np
import sys
import matplotlib.pyplot as plt

from math import *

from datetime import datetime
from mpi4py import MPI
from scipy.linalg import lu_factor, lu_solve

plt.rcParams.update({
    'font.family': 'Arial',   # 设置字体为 Arial
    'font.size': 16           # 字体大小，根据需要修改
})

c=0.00001
cp=1
num=300
yp=np.logspace(start=log(0.01*c), stop=log(105250), num=num, base=np.e)-0.01*c
t=np.logspace(-4,14,num=25000,base=10)

tmin=6.67e-3
idx = np.argmin(np.abs(t - tmin))
Nt=len(t)
zp=np.log(c+yp)

cth=1e-4
w=0.01
K=len(zp)
z_k = zp
dz = np.diff(zp)
Tmatrix=np.zeros([num,num])
slipv=np.zeros(Nt)
slipv[:idx]=1
trac=np.zeros(Nt)
trac[:idx]=1

term=trac[0]*slipv[0]/(cp)

Temp=np.ones(num)*0
T0=0

def Calc_Tmatrix():
    #for j in range(len(self.eleVec)):
    for i in range(K-1):
        if(i==0):
            Tmatrix[0,0]=cth*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
            Tmatrix[0,1]=-cth*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
        else:
            Tmatrix[i,i]=cth*(np.exp(-(2*z_k[i]-dz[i]/2))+np.exp(-(2*z_k[i]+dz[i]/2)))/(dz[i]*dz[i])
            Tmatrix[i,i+1]=-cth*np.exp(-(2*z_k[i]+dz[i]/2))/(dz[i]*dz[i])
            Tmatrix[i,i-1]=-cth*np.exp(-(2*z_k[i]-dz[i]/2))/(dz[i]*dz[i])



def Calc_T_implicit_mpi(Temp1,dt,slipv1,trac1):

    #delta=self.slip
    K=len(zp)
    M=Tmatrix[:-1,:-1]*dt+np.eye(K-1)
    lu, piv = lu_factor(M)
    #print('self.state_local',np.max(self.state_local))

    g=0.0
    #g=-trac1*slipv1/(2.0*cp*cth)

    bv=np.copy(Temp1[:-1])
    bv[0]=bv[0]-2.0*cth*np.exp(-(z_k[0]-dz[0]/2))*g*dt/dz[0]
    #B1.append(b[0])
    bv[-1]=bv[-1]+dt/(dz[-1]*dz[-1])*cth*np.exp(-(2.0*z_k[-2]+dz[-1]/2))*T0
    Ind_term=trac1*slipv1/(cp)*np.exp(-yp[:-1]*yp[:-1]/(2.0*w*w))/(sqrt(2.0*np.pi)*w)*dt
    bv=bv+Ind_term
    
    x = lu_solve((lu, piv), bv)
    Temp1[:-1]=np.copy(x)

    #self.dPdt0[i]=(x[0]*1e-6-self.P[i])/dt
    #term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (Temp[0] - Temp[1]+2*dz[0]*g*exp(z_k[0]))
    #term2 = np.exp(-(z_k[0] + dz[0]/2)) * (Temp[1] - Temp[0])
    #dTdt0[k]=self.cth * np.exp(-z_k[0]) * (term1 + term2) / dz[0]**2+Ind_term[0]
    #print(np.max(np.exp(-self.yp[:-1]*self.yp[:-1]/(2.0*self.halfwidth*self.halfwidth))))
    
    #self.dPdt0[i]=0
    #Temp[k]=Tarr[k,0]
    #print(Tempe[k])
    #print('maxval:',maxval,maxtra)
    #print('rank ',rank,np.max(P[self.local_index]),np.min(P[self.local_index]))
    #return Tempe,dTdt0,Tarr
    return Temp1




# Qana=1.0/(2.0*cth*sqrt(np.pi))*(np.sqrt(4*cth*t+2.0*w*w)-np.sqrt(4*cth*(t-tmin)+2*w*w))

# Qana_tmin=1.0/(2.0*cth*sqrt(np.pi))*(np.sqrt(4*cth*tmin+2.0*w*w)-np.sqrt(4*cth*(tmin-tmin)+2*w*w))
# print(Qana_tmin,term)

# 预因子（不变）
prefactor = 1.0 / (2.0 * cth * np.sqrt(np.pi))

# Qana 的理性化形式
alpha = 4.0 * cth  # 系数
beta = 2.0 * w * w
A = alpha * t + beta
B = alpha * (t - tmin) + beta
diff_ab = alpha * tmin  # a - b = 4 * cth * tmin，常数
Qana = prefactor * (diff_ab / (np.sqrt(A) + np.sqrt(B)))

# Qana_tmin 的理性化形式（t = tmin 的特例）
C = alpha * tmin + beta
Qana_tmin = prefactor * (diff_ab / (np.sqrt(C) + np.sqrt(beta)))  # beta = 2 w^2

Calc_Tmatrix()

Tbound=[]
for i in range(len(t)-1):
    dt=t[i+1]-t[i]
    Temp=Calc_T_implicit_mpi(Temp,dt,slipv[i],trac[i])
    #print(slipv[i],trac[i])
    Tbound.append(Temp[0])

plt.figure(figsize=(10, 6))
plt.plot(t,Qana/Qana_tmin,label='analytical solution', linewidth=1.5)
plt.plot(t[idx:-1],Tbound[idx:]/Qana_tmin,label='numericial solution',linestyle='--', linewidth=2.0)
plt.grid(True, which='both', linestyle='--', linewidth=0.4)

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Time(s)')
plt.ylabel('Q/Qf')
plt.xlim([1e-2,1e14])
plt.savefig('compare_analytical.eps',dpi=500)
plt.show()
