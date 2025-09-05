import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.sparse import coo_matrix
from scipy.interpolate import lagrange
from scipy.interpolate import RegularGridInterpolator
import time
import SH_greenfunction
import DH_greenfunction
from joblib import dump, load
import config
from config import comm, rank, size
#A1=np.load('bp5t_core/A1d.npy')
#A1s=np.load('A1s.npy')
from scipy.sparse.linalg import svds
import sys
from mpi4py import MPI
import ctypes
import pickle
import traceback
import h5py
cp=None



#if(useC==True):
'''Read C++ compiled dynamic library to calculate green's functions'''
try:  
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

    lib.TDstressEachSourceAtReceiver_C.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # x, y, z
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),  # P1, P2, P3
        ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # Ss, Ds, Ts
        ctypes.c_double, ctypes.c_double,  # mu, lambda
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_double),  # stress_out
        ctypes.POINTER(ctypes.c_double)   # strain_out
    ]
except AttributeError as e:
    print(f"Failed to load C dynamic libaray")



TASK_TAG = 1
RESULT_TAG = 2
STOP_TAG = 3  # 结束信号


    
def bounding_box(cluster, points):
    """
    calc bounding box for points
    :param cluster: index of point
    :param points: coordinate,shape (n, d)
    """
    #print(points,cluster)
    a = np.min(points[cluster], axis=0)  # 计算每个维度上的最小值
    b = np.max(points[cluster], axis=0)  # 计算每个维度上的最大值
    return a, b

def diameter(a, b):
    """
    calc bounding box diameter
    :param a: bounding box minimum coord
    :param b: bounding box maximum coord
    :return: v
    """
    return np.sqrt(np.sum((b - a) ** 2))

def distance(a_tau, b_tau, a_sigma, b_sigma):
    """
    calc bounding box distance
    :param a_tau, b_tau: first bounding box minimun and maximum coord
    :param a_sigma, b_sigma: second bounding box minimun and maximum coord
    :return: two bounding box distance
    """
    return np.sqrt(np.sum((np.maximum(0, a_sigma - b_tau) ** 2) + 
                          (np.maximum(0, a_tau - b_sigma) ** 2)))

def is_admissible(cluster_tau,cluster_sigma,points, eta=2.0):
    """
    judge if bounding box meet admissible condition
    :param a_tau, b_tau: first bounding box minimun and maximum coord
    :param a_sigma, b_sigma: second bounding box minimun and maximum coord
    :param eta: admissible condition (通常 0 < η < 3)
    :return: admissibility condition
    """
    a_tau, b_tau = bounding_box(cluster_tau, points)
    a_sigma, b_sigma = bounding_box(cluster_sigma, points)

    d_tau = diameter(a_tau, b_tau)
    d_sigma = diameter(a_sigma, b_sigma)
    d_Q = distance(a_tau, b_tau, a_sigma, b_sigma)
    
    return min(d_tau, d_sigma) <= eta * d_Q


#Singular Value Decomposition
def SVD_recompress(U_ACA, V_ACA, eps):
    """
    Recompress an ACA matrix approximation via SVD.

    :param U_ACA: The left-hand approximation matrix.
    :param V_ACA: The right-hand approximation matrix.
    :param eps: The tolerance of the approximation. The convergence condition is
        in terms of the difference in Frobenius norm between the target matrix
        and the approximation.

    :return U_SVD: The SVD recompressed left-hand approximation matrix.
    :return V_SVD: The SVD recompressed right-hand approximation matrix.
    """
    UQ, UR = np.linalg.qr(U_ACA)
    VQ, VR = np.linalg.qr(V_ACA.T)
    W, SIG, Z = np.linalg.svd(UR.dot(VR.T))

    frob_K = np.sqrt(np.cumsum(SIG[::-1] ** 2))[::-1]
    r = np.argmax(frob_K < eps)

    U = UQ.dot(W[:, :r] * SIG[:r])
    V = Z[:r, :].dot(VQ.T)
    return U, V



class TreeNode:
    """ Binary tree node, BlockTree structure storing H matrix"""
    def __init__(self, indices, split_dim=None, split_value=None,level=0):
        self.indices = indices  # The point index stored in the current node
        self.split_dim = split_dim  # Split dimension
        self.split_value = split_value  # Split threshold
        self.level=level
        self.left = None  # Left subtree
        self.right = None  # Right subtree

def build_block_tree(cluster, points,min_size=16,depth=0):
    """
    Recursively construct BlockTree (binary tree)
    :param cluster: the index set to be divided
    :param points: the coordinate matrix of all points, shape (n, d)
    :param min_size: the minimum size of the cluster, when this value is reached, no more splitting
    :return: the root node
    """
    if len(cluster) <= min_size:
        return TreeNode(cluster,level=depth)  # Leaf node, no longer divided

    d = points.shape[1]  # 维度
    alpha = np.min(points, axis=0)  # Calculate the minimum value of each dimension
    beta = np.max(points, axis=0)   # Calculate the maximum value of each dimension
    #print(alpha,beta)
    # 选择分裂方向，使得 beta_j - alpha_j 最大
    j_max = np.argmax(beta - alpha)
    # 计算分裂阈值
    gamma = (alpha[j_max] + beta[j_max]) / 2

    #index1=np.where(points[:, j_max] <= gamma)[0]
    #index2=np.where(points[:, j_max] > gamma)[0]
    #print(len(index1),len(cluster))
    #tau_1, tau_2 = np.zeros(len(index1)), np.zeros(len(index2))
    index1=points[:, j_max] <= gamma
    index2=points[:, j_max] > gamma
    tau_1=cluster[index1]
    tau_2=cluster[index2]
    #if(depth<=3):
    #    print(depth,len(tau_1),len(tau_2))

    #if ((len(tau_1)< min_size) or (len(tau_2)< min_size)):
    if ((len(tau_1)==0) or (len(tau_2)==0)):
        return TreeNode(cluster,level=depth) 
    
    node = TreeNode(cluster, split_dim=j_max, split_value=gamma,level=depth)
    node.left = build_block_tree(tau_1, points[index1], min_size,depth+1)
    node.right = build_block_tree(tau_2, points[index2], min_size,depth+1)
    
    return node

def print_tree(node, depth=0):
    """ Recursively print the binary tree structure """
    if node is None:
        return
    print(" " * (depth * 2), f"Node: {node.indices}, split_dim={node.split_dim}, split_value={node.split_value}")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)



class Block:
    def __init__(self, row_cluster, col_cluster,row_index,col_index, children=None, level=0):
        """
        Block in HMatrix supports multi-layer recursive block division.
        :param row_cluster: row index
        :param col_cluster: column index
        :param data: matrix data stored in leaf blocks
        :param children: if it is a parent block, it contains a list of child blocks
        :param level: records the level of the current block
        """
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_index=row_index
        self.col_index=col_index
        #self.jud_svd=True
        self.U_1s=[]
        self.S_1s=[]
        self.Vt_1s=[]
        self.U_2s=[]
        self.S_2s=[]
        self.Vt_2s=[]
        self.U_Bs=[]
        self.S_Bs=[]
        self.Vt_Bs=[]

        self.U_1d=[]
        self.S_1d=[]
        self.Vt_1d=[]
        self.U_2d=[]
        self.S_2d=[]
        self.Vt_2d=[]
        self.U_Bd=[]
        self.S_Bd=[]
        self.Vt_Bd=[]

        self.ACA_dictS={}
        self.ACA_dictD={}


        self.Mf_A1s=[]
        self.Mf_A2s=[]
        self.Mf_Bs=[]
        self.Mf_A1d=[]
        self.Mf_A2d=[]
        self.Mf_Bd=[]

        #self.data = data  
        self.children = children if children is not None else []
        self.level = level  
        self.judproc=False
        self.judsvd=False
        self.judaca=False

    def setdata(self,data1):
        self.data=data1

    def is_leaf(self):
        """Judge whether it is a leaf block"""
        return len(self.children) == 0

    def apply_low_rank_approximation(self, rank=10):
        """
        Perform low-rank approximation (SVD decomposition) on leaf blocks
        :param rank: the order of low-rank approximation
        """
        if self.is_leaf() and self.data is not None:
            U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
            k = min(rank, len(S))  # Get the maximum available rank
            self.data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            return self.data
        return None


class BlockTree:
    def __init__(self, root_block,nodelst,elelst,eleVec,mu_,lambda_, xg,halfspace_jud, mini_leaf):
        """HMatrix's BlockTree structure supports multi-layer block division"""
        self.root_block = root_block
        self.nodelst=nodelst
        self.elelst=elelst
        self.eleVec=eleVec
        self.xg=xg
        self.mu_=mu_
        self.lambda_=lambda_
        self.halfspace_jud=halfspace_jud
        #self.Intep_rowindex=[]
        #self.Intep_colindex=[]
        self.count=0
        self.size=0
        self.mini_leaf=mini_leaf
        self.thread_svd=mini_leaf
        self.thread_comm=5000
        self.sparse_csr=[]
        self.yvector=np.zeros(len(xg))
        self.maxdepth=0
        self.progress_tra=0
        self.blocks_to_process=[]
        self.size_local_blocks=0
        
        
        
    #transtate the Stress tensor into traction on the fault surface
    def GetTtstress(self,Stress,col_cluster):
        Tra=[]
        #print(self.eleVec.shape)
        for k in range(col_cluster.shape[0]):
            i=col_cluster[k]
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]

            Tr1=Stress[0,k]*ev31+Stress[3,k]*ev32+Stress[4,k]*ev33
            Tr2=Stress[3,k]*ev31+Stress[1,k]*ev32+Stress[5,k]*ev33
            Trn=Stress[4,k]*ev31+Stress[5,k]*ev32+Stress[2,k]*ev33

            Tt1=Tr1*ev11+Tr2*ev12+Trn*ev13
            Tt2=Tr1*ev21+Tr2*ev22+Trn*ev23
            Tn=Tr1*ev31+Tr2*ev32+Trn*ev33
            Tra.append([Tt1,Tt2,Tn])
        Tra=np.array(Tra)
        return Tra
        
    
        
    #calc strike-slip stressfunc from row in total matrix (calc row elements in ACA)
    def calc_stressfunc_fromC_rowS(self,row_cluster,col_cluster):
        A1s=np.zeros([len(row_cluster),len(col_cluster)])
        A2s=np.zeros([len(row_cluster),len(col_cluster)])
        Bs=np.zeros([len(row_cluster),len(col_cluster)])
        X,Y,Z=self.xg[row_cluster, 0], self.xg[row_cluster, 1], self.xg[row_cluster, 2]
        n=len(X)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Z = np.array(Z, dtype=np.float64)
        
        P1_lst=[]
        P2_lst=[]
        P3_lst=[]
        for i in range(len(col_cluster)):
            I=col_cluster[i]
            P1=np.copy(self.nodelst[self.elelst[I,0]-1])
            P2=np.copy(self.nodelst[self.elelst[I,1]-1])
            P3=np.copy(self.nodelst[self.elelst[I,2]-1])
            P1_lst.append(P1)
            P2_lst.append(P2)
            P3_lst.append(P3)
        n_sources=len(P1_lst)
        P1_lst=np.array(P1_lst).flatten()
        P2_lst=np.array(P2_lst).flatten()
        P3_lst=np.array(P3_lst).flatten()
        useC=config.Para0['Using C++ green function']=='True'
        Ts, Ss, Ds = 0, 1, 0
        for i in range(len(X)):
            if(useC==True):
                stress_out = np.zeros(n_sources * 6, dtype=np.float64)
                strain_out = np.zeros(n_sources * 6, dtype=np.float64)
                lib.TDstressEachSourceAtReceiver_C(
                    X[i], Y[i], Z[i],
                    P1_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    P2_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    P3_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n_sources,
                    Ss, Ds, Ts,
                    self.mu_,self.lambda_,
                    self.halfspace_jud,
                    stress_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    strain_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
                stress_out = stress_out.reshape((n_sources, 6))
                #strain_out = strain_out.reshape((n_sources, 6))
                stress=stress_out.transpose()
            else:
                stress_out = np.zeros([n_sources,6], dtype=np.float64)
                for j in range(len(col_cluster)):
                    I=col_cluster[j]
                    P1=np.copy(self.nodelst[self.elelst[I,0]-1])
                    P2=np.copy(self.nodelst[self.elelst[I,1]-1])
                    P3=np.copy(self.nodelst[self.elelst[I,2]-1])

                    if(self.halfspace_jud==True):
                        stress,strain=SH_greenfunction.TDstressHS(X[i], Y[i], Z[i],P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                    else:
                        stress,strain=SH_greenfunction.TDstressFS(X[i], Y[i], Z[i],P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                    stress_out[j,:]=stress.flatten()
                
                stress=stress_out.transpose()
            
            #col_strans= np.tile(row_cluster[i], (n_sources, 1))
            col_strans = np.full(n_sources,row_cluster[i])
            #print(col_strans.shape)
            Tra = self.GetTtstress(stress,col_strans)
            # if(np.max(np.abs(Tra))>1e15):
            #     print(np.max(Tra),'!!!!!!!!!!!!!!!!!!!!!!!!calc_stressfunc_fromC_rowS')
            #     MPI.Finalize()
            #     sys.exit()
            A1s[i]=Tra[:, 0]
            A2s[i]=Tra[:, 1]
            Bs[i]=Tra[:, 2]
        return A1s,A2s,Bs
    
    #calc dip-slip stressfunc from row in total matrix (calc row elements in ACA)
    def calc_stressfunc_fromC_rowD(self,row_cluster,col_cluster):
        A1d=np.zeros([len(row_cluster),len(col_cluster)])
        A2d=np.zeros([len(row_cluster),len(col_cluster)])
        Bd=np.zeros([len(row_cluster),len(col_cluster)])
        X,Y,Z=self.xg[row_cluster, 0], self.xg[row_cluster, 1], self.xg[row_cluster, 2]
        n=len(X)
        
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Z = np.array(Z, dtype=np.float64)
        P1_lst=[]
        P2_lst=[]
        P3_lst=[]
        for i in range(len(col_cluster)):
            I=col_cluster[i]
            P1=np.copy(self.nodelst[self.elelst[I,0]-1])
            P2=np.copy(self.nodelst[self.elelst[I,1]-1])
            P3=np.copy(self.nodelst[self.elelst[I,2]-1])
            P1_lst.append(P1)
            P2_lst.append(P2)
            P3_lst.append(P3)
        n_sources=len(P1_lst)
        P1_lst=np.array(P1_lst).flatten()
        P2_lst=np.array(P2_lst).flatten()
        P3_lst=np.array(P3_lst).flatten()
        
        
        Ts, Ss, Ds = 0, 0, 1
        useC=config.Para0['Using C++ green function']=='True'
        #useC=True
        for i in range(len(X)):
            if(useC==True):
                stress_out = np.zeros(n_sources * 6, dtype=np.float64)
                strain_out = np.zeros(n_sources * 6, dtype=np.float64)
                lib.TDstressEachSourceAtReceiver_C(
                    X[i], Y[i], Z[i],
                    P1_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    P2_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    P3_lst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n_sources,
                    Ss, Ds, Ts,
                    self.mu_,self.lambda_,
                    self.halfspace_jud,
                    stress_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    strain_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
                stress_out = stress_out.reshape((n_sources, 6))
                #strain_out = strain_out.reshape((n_sources, 6))
                stress=stress_out.transpose()
            else:
                stress_out = np.zeros([n_sources,6], dtype=np.float64)
                for j in range(len(col_cluster)):
                    I=col_cluster[j]
                    P1=np.copy(self.nodelst[self.elelst[I,0]-1])
                    P2=np.copy(self.nodelst[self.elelst[I,1]-1])
                    P3=np.copy(self.nodelst[self.elelst[I,2]-1])

                    if(self.halfspace_jud==True):
                        stress,strain=SH_greenfunction.TDstressHS(X[i], Y[i], Z[i],P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                    else:
                        stress,strain=SH_greenfunction.TDstressFS(X[i], Y[i], Z[i],P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                    stress_out[j,:]=stress.flatten()
                stress=stress_out.transpose()
            
            col_strans = np.full(n_sources,row_cluster[i])
            Tra = self.GetTtstress(stress,col_strans)
            A1d[i]=Tra[:, 0]
            A2d[i]=Tra[:, 1]
            Bd[i]=Tra[:, 2]
        return A1d,A2d,Bd
        
    #calc strike-slip stressfunc from column in total matrix (calc column elements in ACA)
    def calc_stressfunc_fromCS(self,row_cluster,col_cluster):
        A1s=np.zeros([len(row_cluster),len(col_cluster)])
        A2s=np.zeros([len(row_cluster),len(col_cluster)])
        Bs=np.zeros([len(row_cluster),len(col_cluster)])
        X,Y,Z=self.xg[row_cluster, 0], self.xg[row_cluster, 1], self.xg[row_cluster, 2]
        
        n=len(X)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Z = np.array(Z, dtype=np.float64)
        #print(X)
        Ts, Ss, Ds = 0, 1, 0
        for i in range(len(col_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[col_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[col_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[col_cluster[i],2]-1])
            
            # if(self.halfspace_jud==True):
            #     stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            # else:
                #stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)

            useC=config.Para0['Using C++ green function']=='True'
            #useC=True
            if(useC==True):
                stress = np.zeros(n * 6)
                strain = np.zeros(n * 6)
                lib.TDstressFS_C(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            n,
                            P1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            P2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            P3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Ss,Ds,Ts,   # Ss, Ds, Ts
                            self.mu_,self.lambda_,       # mu, lambda
                            self.halfspace_jud,
                            stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            strain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                            )
                stress=stress.reshape([n,6])
                stress=stress.transpose()
            else:
                if(self.halfspace_jud==True):
                    stress,strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                else:
                    stress,strain=SH_greenfunction.TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            
            #print(stress)
            Tra = self.GetTtstress(stress,row_cluster)
            # if(np.max(np.abs(Tra))>1e15):
            #     print(np.max(Tra),'!!!!!!!!!!!!!!!!!!!!!!!!calc_stressfunc_fromCS')
            #     MPI.Finalize()
            #     sys.exit()
            A1s[:,i]=Tra[:, 0]
            A2s[:,i]=Tra[:, 1]
            Bs[:,i]=Tra[:, 2]
        
        
        return A1s,A2s,Bs
        
    #calc dip-slip stressfunc from column in total matrix (calc column elements in ACA)    
    def calc_stressfunc_fromCD(self,row_cluster,col_cluster):
        A1d=np.zeros([len(row_cluster),len(col_cluster)])
        A2d=np.zeros([len(row_cluster),len(col_cluster)])
        Bd=np.zeros([len(row_cluster),len(col_cluster)])
        X,Y,Z=self.xg[row_cluster, 0], self.xg[row_cluster, 1], self.xg[row_cluster, 2]
        n=len(X)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Z = np.array(Z, dtype=np.float64)
        Ts, Ss, Ds = 0, 0, 1
        for i in range(len(col_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[col_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[col_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[col_cluster[i],2]-1])
            # if(self.halfspace_jud==True):
            #     Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            #else:
                #Stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            useC=config.Para0['Using C++ green function']=='True'
            #useC=True
            if(useC==True):
                stress = np.zeros(n * 6)
                strain = np.zeros(n * 6)
                lib.TDstressFS_C(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            n,
                            P1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            P2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            P3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            Ss,Ds,Ts,   # Ss, Ds, Ts
                            self.mu_,self.lambda_,       # mu, lambda
                            self.halfspace_jud,
                            stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            strain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                            )
                stress=stress.reshape([n,6])
                stress=stress.transpose()
            else:
                print(useC,'useC==False..!!!!!!!!!!')
                if(self.halfspace_jud==True):
                    stress,strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
                else:
                    stress,strain=SH_greenfunction.TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            
            
            Tra = self.GetTtstress(stress,row_cluster)
            A1d[:,i]=Tra[:, 0]
            A2d[:,i]=Tra[:, 1]
            Bd[:,i]=Tra[:, 2]
        return A1d,A2d,Bd
    

    def calc_stressfunc(self,row_cluster,col_cluster):
        A1s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bs=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A1d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bd=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        X,Y,Z=self.xg[col_cluster, 0], self.xg[col_cluster, 1], self.xg[col_cluster, 2]

        Ts, Ss, Ds = 0, 1, 0
        for i in range(len(row_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[row_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[row_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[row_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=SH_greenfunction.TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,col_cluster)
            A1s[i]=Tra[:, 0]
            A2s[i]=Tra[:, 1]
            Bs[i]=Tra[:, 2]

        Ts, Ss, Ds = 0, 0, 1
        for i in range(len(row_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[row_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[row_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[row_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=SH_greenfunction.TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,col_cluster)
            A1d[i]=Tra[:, 0]
            A2d[i]=Tra[:, 1]
            Bd[i]=Tra[:, 2]
        return A1s.transpose(),A2s.transpose(),Bs.transpose(),A1d.transpose(),A2d.transpose(),Bd.transpose()
    
    
    #Use A(i, j) to construct a row extraction function
    def calc_rowsS(self,Istart, Iend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        A1s,A2s,Bs=self.calc_stressfunc_fromC_rowS(row_cluster[Istart: Iend],col_cluster)
        #A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(row_cluster[Istart: Iend],col_cluster)
        return A1s,A2s,Bs
        
    def calc_rowsD(self,Istart, Iend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        A1d,A2d,Bd=self.calc_stressfunc_fromC_rowD(row_cluster[Istart: Iend],col_cluster)
        #A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(row_cluster[Istart: Iend],col_cluster)
        return A1d,A2d,Bd
    
    def calc_colsS(self,Jstart, Jend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        #A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(row_cluster,col_cluster[Jstart: Jend])
        A1s,A2s,Bs=self.calc_stressfunc_fromCS(row_cluster,col_cluster[Jstart: Jend])
        return A1s,A2s,Bs
        
    def calc_colsD(self,Jstart, Jend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        #A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(row_cluster,col_cluster[Jstart: Jend])
        A1d,A2d,Bd=self.calc_stressfunc_fromCD(row_cluster,col_cluster[Jstart: Jend])
        return A1d,A2d,Bd
        
    #ACA calculation for strike-slip green functions
    def ACA_plus_full_traceS(self,n_rows, n_cols, calc_rows, calc_cols, row_cluster,col_cluster,eps, max_iter=None, verbose=False):
        us_A1s,vs_A1s=[],[]
        us_A2s,vs_A2s=[],[]
        us_Bs,vs_Bs=[],[]
        #start_time = time.time()

        prevIstar, prevJstar = [], []


        def argmax_not_in_list(arr, disallowed):
            arg_sorted = arr.argsort()
            max_idx = arg_sorted.shape[0] - 1
            while max_idx >= 0:
                if arg_sorted[max_idx] not in disallowed:
                    return arg_sorted[max_idx]
                max_idx -= 1
            raise RuntimeError("All entries disallowed.")

        def calc_residual_rows(Istart, Iend):
            A1s_out,A2s_out,Bs_out = calc_rows(Istart, Iend,row_cluster,col_cluster)
            #print(A1s_out.shape,row_cluster.shape,col_cluster.shape)
            for i in range(len(us_A1s)):
                A1s_out -= us_A1s[i][Istart:Iend][:, None] * vs_A1s[i][None, :]
                A2s_out -= us_A2s[i][Istart:Iend][:, None] * vs_A2s[i][None, :]
                Bs_out -= us_Bs[i][Istart:Iend][:, None] * vs_Bs[i][None, :]
            return A1s_out,A2s_out,Bs_out

        def calc_residual_cols(Jstart, Jend):
            A1s_out,A2s_out,Bs_out = calc_cols(Jstart, Jend, row_cluster,col_cluster)
            for i in range(len(us_A1s)):
                A1s_out -= vs_A1s[i][Jstart:Jend][None,:] * us_A1s[i][:, None]
                A2s_out -= vs_A2s[i][Jstart:Jend][None,:] * us_A2s[i][:, None]
                Bs_out -= vs_Bs[i][Jstart:Jend][None,:] * us_Bs[i][:, None]
            return A1s_out,A2s_out,Bs_out


        def reset_reference_row(Iref):
            attemp=0
            max_attempts = n_rows // 3 + 1 
            scan_all=False
            while True:
                Iref = (Iref + 3) % n_rows
                Iref -= Iref % 3
                attemp=attemp+1
                if Iref not in prevIstar:
                    break
                if(attemp>max_attempts):
                    scan_all=True
                    break
            out,out1,out2=calc_residual_rows(Iref, Iref + 3)
            return out,out1,out2, Iref,scan_all

        def reset_reference_col(Jref):
            attemp=0
            max_attempts = n_cols // 3 + 1 
            scan_all=False
            while True:
                Jref = (Jref + 3) % n_cols
                Jref -= Jref % 3
                attemp=attemp+1
                if Jref not in prevJstar:
                    break
                if(attemp>max_attempts):
                    scan_all=True
                    break
            out,out1,out2=calc_residual_cols(Jref, Jref + 3)
            return out,out1,out2, Jref,scan_all

        if max_iter is None:
            max_iter = min(n_rows, n_cols)

        RIstar = np.zeros(n_cols)
        RJstar = np.zeros(n_rows)
        RIstar1 = np.zeros(n_cols)
        RJstar1 = np.zeros(n_rows)
        RIstar2 = np.zeros(n_cols)
        RJstar2 = np.zeros(n_rows)


        Iref = np.random.randint(n_rows) - 3
        Jref = np.random.randint(n_cols) - 3
        RIref,RIref1,RIref2,Iref,scan_all = reset_reference_row(Iref)
        RJref,RJref1,RJref2,Jref,scan_all = reset_reference_col(Jref)
        iter=0
        for k in range(max_iter):
            maxabsRIref = np.max(np.abs(RIref), axis=0)
            Jstar = argmax_not_in_list(maxabsRIref, prevJstar)

            maxabsRJref = np.max(np.abs(RJref), axis=1)
            Istar = argmax_not_in_list(maxabsRJref, prevIstar)

            Jstar_val = maxabsRIref[Jstar]
            Istar_val = maxabsRJref[Istar]
            #print(f"!!!!!!!!!!!!!!!!!!!!1",flush=True)
            if Istar_val > Jstar_val:
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
                Jstar = argmax_not_in_list(np.abs(RIstar), prevJstar)
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
            else:
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
                Istar = argmax_not_in_list(np.abs(RJstar), prevIstar)
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
            #print(f"!!!!!!!!!!!!!!!!!!!!2",flush=True)
            alpha = RIstar[Jstar]
            alpha1= RIstar1[Jstar]
            alpha2 = RIstar2[Jstar]

            

            prevIstar.append(Istar)
            prevJstar.append(Jstar)

            u_k = RJstar.copy()
            v_k = RIstar.copy()
            u_k1 = RJstar1.copy()
            v_k1 = RIstar1.copy()
            u_k2 = RJstar2.copy()
            v_k2 = RIstar2.copy()

            #print(f"!!!!!!!!!!!!!!!!!!!!3",flush=True)
            us_A1s.append(u_k)
            vs_A1s.append(v_k/(alpha+1e-16))
            #print('alpha',alpha)
            us_A2s.append(u_k1)
            vs_A2s.append(v_k1/(alpha1+1e-16))
            us_Bs.append(u_k2)
            if(abs(alpha2)<1e-10):
                vs_Bs.append(np.zeros(len(v_k2)))
            else:
                vs_Bs.append(v_k2/(alpha2+1e-16))
            

            step_size = np.sqrt(np.sum(u_k ** 2) * np.sum((v_k / (alpha+1e-10)) ** 2))
            #step_size1 = np.sqrt(np.sum(u_k1 ** 2) * np.sum((v_k1 / (alpha1+1e-10)) ** 2))
            iter=k
            #step_size=np.max([step_size,step_size1])

            if verbose:
                print(
                    f"Siteration:{k},pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                    f"step size={step_size:1.13e}, "
                    f"np.abs(alpha)={np.abs(alpha):1.13e}, "
                    f"tolerance={eps:1.13e}", flush=True
                )

           
            #print(f"!!!!!!!!!!!!!!!!!!!!__",flush=True)
            if Iref <= Istar < Iref + 3:
                #print('1')
                RIref,RIref1,RIref2,Iref,scan_all = reset_reference_row(Iref)
            else:
                #print('2')
                RIref -= u_k[Iref:Iref+3][:, None] * (v_k / alpha)[None, :]
            if(scan_all==True):
                break
            
            #print(f"!!!!!!!!!!!!!!!!!!!!_",flush=True)
            if Jref <= Jstar < Jref + 3:
                #print('1')
                RJref,RJref1,RJref2,Jref,scan_all = reset_reference_col(Jref)
                #print(RJref,Jref)
            else:
                #print('2')
                RJref -= (v_k / alpha)[Jref:Jref+3][None, :] * u_k[:, None]
            if(scan_all==True):
                break
            
            if np.abs(alpha) < 1e-4:
                if verbose:
                    print(f"Terminated at k={k} due to small pivot.",alpha)
                #sys.exit()
                break

            if step_size < eps:
                break
            
            if k == max_iter - 1:
                break
        

        U_ACA_A1s = np.array(us_A1s).T
        V_ACA_A1s = np.array(vs_A1s)

        us_Bs=np.array(us_Bs)
        if not np.all(us_Bs == 0):
            U_ACA_Bs = us_Bs.T
            V_ACA_Bs = np.array(vs_Bs)
        else:
            U_ACA_Bs = []
            V_ACA_Bs = []
        
        #if(step_size1>eps):
        
        us_A1s,vs_A1s=[],[]
        us_A2s,vs_A2s=[],[]
        us_Bs,vs_Bs=[],[]

        prevIstar, prevJstar = [], []  
        for k in range(max_iter):
            maxabsRIref = np.max(np.abs(RIref1), axis=0)
            Jstar = argmax_not_in_list(maxabsRIref, prevJstar)
            maxabsRJref = np.max(np.abs(RJref1), axis=1)
            Istar = argmax_not_in_list(maxabsRJref, prevIstar)
            Jstar_val = maxabsRIref[Jstar]
            Istar_val = maxabsRJref[Istar]
            
            #print(f"!!!!!!!!!!!!!!!!!!!!1",flush=True)
            if Istar_val > Jstar_val:
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
                Jstar = argmax_not_in_list(np.abs(RIstar1), prevJstar)
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
            else:
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
                Istar = argmax_not_in_list(np.abs(RJstar1), prevIstar)
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
            #print(f"!!!!!!!!!!!!!!!!!!!!2",flush=True)
            alpha = RIstar[Jstar]
            alpha1= RIstar1[Jstar]
            alpha2 = RIstar2[Jstar]

            if np.abs(alpha1) < 1e-4:
                if verbose:
                    print(f"Terminated at k={k} due to small pivot.",alpha1)
                #sys.exit()
                break

            prevIstar.append(Istar)
            prevJstar.append(Jstar)


            u_k = RJstar.copy()
            v_k = RIstar.copy()
            u_k1 = RJstar1.copy()
            v_k1 = RIstar1.copy()
            u_k2 = RJstar2.copy()
            v_k2 = RIstar2.copy()

            #print(f"!!!!!!!!!!!!!!!!!!!!3",flush=True)
            us_A1s.append(u_k)
            vs_A1s.append(v_k/(alpha+1e-16))
            #print('alpha',alpha)
            us_A2s.append(u_k1)
            vs_A2s.append(v_k1/(alpha1+1e-16))
            us_Bs.append(u_k2)
            if(abs(alpha2)<1e-10):
                vs_Bs.append(np.zeros(len(v_k2)))
            else:
                vs_Bs.append(v_k2/(alpha2+1e-16))
            

            #step_size = np.sqrt(np.sum(u_k ** 2) * np.sum((v_k / (alpha+1e-10)) ** 2))
            step_size1 = np.sqrt(np.sum(u_k1 ** 2) * np.sum((v_k1 / (alpha1+1e-10)) ** 2))
            iter=k
            #step_size=np.max([step_size,step_size1])

            if verbose:
                print(
                    f"phase2..!!!!!!!!!!,"
                    f"Siteration:{k},pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                    f"step size1={step_size1:1.13e}, "
                    f"np.abs(alpha1)={np.abs(alpha1):1.13e}, "
                    f"tolerance={eps:1.13e}", flush=True
                )

            if step_size1 < eps:
                break
            
            if k == max_iter - 1:
                break
            #print(f"!!!!!!!!!!!!!!!!!!!!__",flush=True)
            if Iref <= Istar < Iref + 3:
                #print('1')
                RIref,RIref1,RIref2,Iref,scan_all = reset_reference_row(Iref)
            else:
                #print('2')
                RIref -= u_k1[Iref:Iref+3][:, None] * (v_k1 / (alpha1+1e-10))[None, :]
            if(scan_all==True):
                break
            
            #print(f"!!!!!!!!!!!!!!!!!!!!_",flush=True)
            if Jref <= Jstar < Jref + 3:
                #print('1')
                RJref,RJref1,RJref2,Jref,scan_all = reset_reference_col(Jref)
                #print(RJref,Jref)
            else:
                #print('2')
                RJref -= (v_k1 / alpha1)[Jref:Jref+3][None, :] * u_k1[:, None]
            if(scan_all==True):
                break


        U_ACA_A2s = np.array(us_A2s).T
        V_ACA_A2s = np.array(vs_A2s)



        # U_ACA_A2s[U_ACA_A2s>1e15]=0
        # V_ACA_A2s[V_ACA_A2s>1e15]=0
        # if(len(U_ACA_A2s)>0):
        #     if(np.max(np.abs(U_ACA_A2s))>1e20):
                
        #         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!',np.max(np.abs(U_ACA_A1s)),np.max(np.abs(U_ACA_A2s)))
        #         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!',step_size,step_size1,iter,scan_all)
        #         #MPI.Finalize()
        #         sys.exit()
        
        

        
       

        trace_data = {
            "U_ACA_A1s": U_ACA_A1s,
            "V_ACA_A1s": V_ACA_A1s,
            "U_ACA_A2s": U_ACA_A2s,
            "V_ACA_A2s": V_ACA_A2s,
            "U_ACA_Bs": U_ACA_Bs,
            "V_ACA_Bs": V_ACA_Bs,
        }

        return trace_data
    
    #ACA calculation for dip-slip green functions
    def ACA_plus_full_traceD(self,n_rows, n_cols, calc_rows, calc_cols, row_cluster,col_cluster,eps, max_iter=None, verbose=False):
        us_A1d,vs_A1d=[],[]
        us_A2d,vs_A2d=[],[]
        us_Bd,vs_Bd=[],[]
        prevIstar, prevJstar = [], []


        def argmax_not_in_list(arr, disallowed):
            arg_sorted = arr.argsort()
            max_idx = arg_sorted.shape[0] - 1
            while max_idx >= 0:
                if arg_sorted[max_idx] not in disallowed:
                    return arg_sorted[max_idx]
                max_idx -= 1
            raise RuntimeError("All entries disallowed.")

        def calc_residual_rows(Istart, Iend):
            A1d_out,A2d_out,Bd_out = calc_rows(Istart, Iend,row_cluster,col_cluster)
            #print(A1s_out.shape,row_cluster.shape,col_cluster.shape)
            for i in range(len(us_A1d)):
                A1d_out -= us_A1d[i][Istart:Iend][:, None] * vs_A1d[i][None, :]
                A2d_out -= us_A2d[i][Istart:Iend][:, None] * vs_A2d[i][None, :]
                Bd_out -= us_Bd[i][Istart:Iend][:, None] * vs_Bd[i][None, :]
            return A1d_out,A2d_out,Bd_out

        def calc_residual_cols(Jstart, Jend):
            A1d_out,A2d_out,Bd_out = calc_cols(Jstart, Jend, row_cluster,col_cluster)
            for i in range(len(us_A1d)):
                A1d_out -= vs_A1d[i][Jstart:Jend][None,:] * us_A1d[i][:, None]
                A2d_out -= vs_A2d[i][Jstart:Jend][None,:] * us_A2d[i][:, None]
                Bd_out -= vs_Bd[i][Jstart:Jend][None,:] * us_Bd[i][:, None]
            return A1d_out,A2d_out,Bd_out

        def reset_reference_row(Iref):
            attemp=0
            max_attempts = n_rows // 3 + 1 
            scan_all=False
            while True:
                Iref = (Iref + 3) % n_rows
                Iref -= Iref % 3
                attemp=attemp+1
                if Iref not in prevIstar:
                    break
                if(attemp>max_attempts):
                    scan_all=True
                    break
            out,out1,out2=calc_residual_rows(Iref, Iref + 3)
            return out,out1,out2, Iref,scan_all

        def reset_reference_col(Jref):
            attemp=0
            max_attempts = n_cols // 3 + 1 
            scan_all=False
            while True:
                Jref = (Jref + 3) % n_cols
                Jref -= Jref % 3
                attemp=attemp+1
                if Jref not in prevJstar:
                    break
                if(attemp>max_attempts):
                    scan_all=True
                    break
            out,out1,out2=calc_residual_cols(Jref, Jref + 3)
            return out,out1,out2, Jref,scan_all


        if max_iter is None:
            max_iter = min(n_rows, n_cols)

        RIstar = np.zeros(n_cols)
        RJstar = np.zeros(n_rows)
        RIstar1 = np.zeros(n_cols)
        RJstar1 = np.zeros(n_rows)
        RIstar2 = np.zeros(n_cols)
        RJstar2 = np.zeros(n_rows)


        Iref = np.random.randint(n_rows) - 3
        Jref = np.random.randint(n_cols) - 3
        RIref,RIref1,RIref2,Iref,scan_all = reset_reference_row(Iref)
        RJref,RJref1,RJref2,Jref,scan_all = reset_reference_col(Jref)
        
        for k in range(max_iter):
            maxabsRIref = np.max(np.abs(RIref), axis=0)
            Jstar = argmax_not_in_list(maxabsRIref, prevJstar)
            
            maxabsRJref = np.max(np.abs(RJref), axis=1)
            Istar = argmax_not_in_list(maxabsRJref, prevIstar)
            #print(f"!!!!!!!!!!!!!!!!!!!!1",flush=True)
            Jstar_val = maxabsRIref[Jstar]
            Istar_val = maxabsRJref[Istar]

            if Istar_val > Jstar_val:
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
                Jstar = argmax_not_in_list(np.abs(RIstar), prevJstar)
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
            else:
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
                Istar = argmax_not_in_list(np.abs(RJstar), prevIstar)
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
            #print(f"!!!!!!!!!!!!!!!!!!!!2",flush=True)
            alpha3 = RIstar[Jstar]
            alpha4= RIstar1[Jstar]
            alpha5 = RIstar2[Jstar]

            

            prevIstar.append(Istar)
            prevJstar.append(Jstar)
            #Istar_list.append(Istar)
            #Jstar_list.append(Jstar)

            #print(f"!!!!!!!!!!!!!!!!!!!!3",flush=True)
            u_k3 = RJstar.copy()
            v_k3 = RIstar.copy()
            u_k4 = RJstar1.copy()
            v_k4 = RIstar1.copy()
            u_k5 = RJstar2.copy()
            v_k5 = RIstar2.copy()


            us_A1d.append(u_k3)
            vs_A1d.append(v_k3/(alpha3+1e-16))
            us_A2d.append(u_k4)
            vs_A2d.append(v_k4/(alpha4+1e-16))
            us_Bd.append(u_k5)
            if(abs(alpha5)<1e-10):
                vs_Bd.append(np.zeros(len(v_k5)))
            else:
                vs_Bd.append(v_k5/(alpha5+1e-16))

            step_size = np.sqrt(np.sum(u_k3 ** 2) * np.sum((v_k3 / alpha3) ** 2))
            step_size1 = np.sqrt(np.sum(u_k4 ** 2) * np.sum((v_k4 / alpha4) ** 2))
            if verbose:
                print(
                    f"Diteration:{k},pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                    f"step size={step_size:1.3e}, "
                    f"tolerance={eps:1.3e}", flush=True
                )
            

            if k == max_iter - 1:
                break

            if Iref <= Istar < Iref + 3:
                RIref,RIref1,RIref2, Iref,scan_all = reset_reference_row(Iref)
            else:
                RIref -= u_k3[Iref:Iref+3][:, None] * (v_k3 / alpha3)[None, :]
            #print(f"!!!!!!!!!!!!!!!!!!!!_",flush=True)
            if(scan_all==True):
                break

            if Jref <= Jstar < Jref + 3:
                RJref,RJref1,RJref2, Jref,scan_all = reset_reference_col(Jref)
            else:
                RJref -= (v_k3 / alpha3)[Jref:Jref+3][None, :] * u_k3[:, None]
            #print(f"!!!!!!!!!!!!!!!!!!!!0",flush=True)
            if(scan_all==True):
                break

            if np.abs(alpha3) < 1e-4:
                if verbose:
                    print(f"Terminated at k={k} due to small pivot.")
                break
            
            if step_size < eps:
                break
        
        U_ACA_A1d = np.array(us_A1d).T
        V_ACA_A1d = np.array(vs_A1d)
        us_Bd=np.array(us_Bd)
        if not np.all(us_Bd == 0):
            U_ACA_Bd = us_Bd.T
            V_ACA_Bd = np.array(vs_Bd)
        else:
            U_ACA_Bd = []
            V_ACA_Bd = []
        
        us_A1d,vs_A1d=[],[]
        us_A2d,vs_A2d=[],[]
        us_Bd,vs_Bd=[],[]
        prevIstar, prevJstar = [], []
        
        for k in range(max_iter):
            maxabsRIref = np.max(np.abs(RIref1), axis=0)
            Jstar = argmax_not_in_list(maxabsRIref, prevJstar)
            
            maxabsRJref = np.max(np.abs(RJref1), axis=1)
            Istar = argmax_not_in_list(maxabsRJref, prevIstar)
            #print(f"!!!!!!!!!!!!!!!!!!!!1",flush=True)
            Jstar_val = maxabsRIref[Jstar]
            Istar_val = maxabsRJref[Istar]

            if Istar_val > Jstar_val:
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
                Jstar = argmax_not_in_list(np.abs(RIstar1), prevJstar)
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
            else:
                out,out1,out2 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:]=out[:, 0],out1[:, 0],out2[:, 0]
                Istar = argmax_not_in_list(np.abs(RJstar1), prevIstar)
                out,out1,out2 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:]=out[0],out1[0],out2[0]
            #print(f"!!!!!!!!!!!!!!!!!!!!2",flush=True)
            alpha3 = RIstar[Jstar]
            alpha4= RIstar1[Jstar]
            alpha5 = RIstar2[Jstar]

            

            prevIstar.append(Istar)
            prevJstar.append(Jstar)
            #Istar_list.append(Istar)
            #Jstar_list.append(Jstar)

            #print(f"!!!!!!!!!!!!!!!!!!!!3",flush=True)
            u_k3 = RJstar.copy()
            v_k3 = RIstar.copy()
            u_k4 = RJstar1.copy()
            v_k4 = RIstar1.copy()
            u_k5 = RJstar2.copy()
            v_k5 = RIstar2.copy()


            us_A1d.append(u_k3)
            vs_A1d.append(v_k3/(alpha3+1e-16))
            us_A2d.append(u_k4)
            vs_A2d.append(v_k4/(alpha4+1e-16))
            us_Bd.append(u_k5)
            if(abs(alpha5)<1e-10):
                vs_Bd.append(np.zeros(len(v_k5)))
            else:
                vs_Bd.append(v_k5/(alpha5+1e-16))

            step_size = np.sqrt(np.sum(u_k3 ** 2) * np.sum((v_k3 / (alpha3+1e-10)) ** 2))
            step_size1 = np.sqrt(np.sum(u_k4 ** 2) * np.sum((v_k4 / (alpha4+1e-10)) ** 2))
            if verbose:
                print(
                    f"Diteration:{k},pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                    f"step size={step_size:1.3e}, "
                    f"tolerance={eps:1.3e}", flush=True
                )
            

            if k == max_iter - 1:
                break

            if Iref <= Istar < Iref + 3:
                RIref,RIref1,RIref2, Iref,scan_all = reset_reference_row(Iref)
            else:
                RIref -= u_k4[Iref:Iref+3][:, None] * (v_k4 / (alpha4+1e-10))[None, :]
            #print(f"!!!!!!!!!!!!!!!!!!!!_",flush=True)
            if(scan_all==True):
                break

            if Jref <= Jstar < Jref + 3:
                RJref,RJref1,RJref2, Jref,scan_all = reset_reference_col(Jref)
            else:
                RJref -= (v_k4 / alpha4)[Jref:Jref+3][None, :] * u_k4[:, None]
            #print(f"!!!!!!!!!!!!!!!!!!!!0",flush=True)
            if(scan_all==True):
                break

            if np.abs(alpha4) < 1e-4:
                if verbose:
                    print(f"Terminated at k={k} due to small pivot.")
                break
            
            if step_size1 < eps:
                break
            

        

        U_ACA_A2d = np.array(us_A2d).T
        V_ACA_A2d = np.array(vs_A2d)


        
        
       

        trace_data = {
            "U_ACA_A1d": U_ACA_A1d,
            "V_ACA_A1d": V_ACA_A1d,
            "U_ACA_A2d": U_ACA_A2d,
            "V_ACA_A2d": V_ACA_A2d,
            "U_ACA_Bd": U_ACA_Bd,
            "V_ACA_Bd": V_ACA_Bd
        }

        return trace_data
        
    #Perform ACA calculations in each rank
    def ACA_worker(self,block):
        #block.Mf_Bd=[1,2,3,4,5]
        #If the submatrix is ​​larger than the minimum number of leaves, it means admissible
        if(len(block.row_cluster) > self.thread_svd and len(block.col_cluster) > self.thread_svd):
            print(f'rank {rank},start ACA...', flush=True)
            block.judaca=True
            eps=1.0
            start_time = time.time()
            ACA_dictS=self.ACA_plus_full_traceS(
                len(block.row_cluster), len(block.col_cluster), 
                self.calc_rowsS, 
                self.calc_colsS, 
                block.row_cluster,
                block.col_cluster,
                eps/100.0,
                #max_iter=500,
                verbose=False)
            
            ACA_dictD=self.ACA_plus_full_traceD(
                len(block.row_cluster), len(block.col_cluster), 
                self.calc_rowsD, 
                self.calc_colsD, 
                block.row_cluster,
                block.col_cluster,
                eps/100.0,
                #max_iter=500,
                verbose=False)
            
           
            block.ACA_dictS=ACA_dictS
            block.ACA_dictD=ACA_dictD
            end_time = time.time()
            #print(f"Green func calc_stressfunc_fromC Time taken: {end_time - start_time:.10f} seconds")
            print(f"rank {rank},ACA of task success,size{len(block.row_cluster),len(block.col_cluster)},Time takes {end_time - start_time:.10f} seconds", flush=True)
            
            self.size += (block.ACA_dictS['U_ACA_A1s'].nbytes + block.ACA_dictS['V_ACA_A1s'].nbytes)*6.0/(1024*1024)
        else: #Non-admissible case, to calculate dense sub-matrix
            print(f'rank {rank},start fullmatrix calc...', flush=True)
            start_time = time.time()
            A1s,A2s,Bs=self.calc_stressfunc_fromCS(block.row_cluster,block.col_cluster)
            A1d,A2d,Bd=self.calc_stressfunc_fromCD(block.row_cluster,block.col_cluster)
            block.Mf_A1s=A1s
            block.Mf_A2s=A2s
            block.Mf_Bs=Bs
            block.Mf_A1d=A1d
            block.Mf_A2d=A2d
            block.Mf_Bd=Bd
            self.size +=A1s.nbytes*6/(1024*1024)
            end_time = time.time()
            print(f"rank {rank},Full matrix calculation task success,size{len(block.row_cluster),len(block.col_cluster)},Time takes {end_time - start_time:.10f} seconds", flush=True)
        block.judproc=True
        
        return block
    
    
    
    def master_scatter(self,dir,blocks_to_process,num_workers):
        
        def print_progress(current, total):
            bar_len = 30
            filled_len = int(round(bar_len * current / float(total)))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write(f'\rProgress: [{bar}] {current}/{total} results collected')
            sys.stdout.flush()
        # 主进程准备任务
        if rank == 0:
            task_id = 1
            next_task = 0
            #nprocs = num_workers
            
            #print('label',len(self.Label))
            if(len(blocks_to_process)==0):
                self.collect_blocks(self.root_block)
                #self.blocks_to_process.sort(key=lambda block: len(block.row_cluster), reverse=True)
            else:
                self.blocks_to_process=blocks_to_process

            # 按进程数量平均切分任务
            def split_blocks(blocks, nprocs):
                n = len(blocks)
                q, r = divmod(n, nprocs)
                chunks = []
                start = 0
                for i in range(nprocs):
                    end = start + q + (1 if i < r else 0)
                    chunks.append(blocks[start:end])
                    start = end
                return chunks

            task_chunks = split_blocks(self.blocks_to_process, num_workers)
        else:
            task_chunks = None

        # 分发任务：每个进程获得一小块任务
        my_blocks = comm.scatter(task_chunks, root=0)

        # 每个进程处理自己的任务
        #my_results = [self.ACA_worker(block) for block in my_blocks]
        my_results = []
        nblocks = len(my_blocks)
        for i, block in enumerate(my_blocks):
            try:
                result = self.ACA_worker(block)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Rank {rank}] ❌ Exception on block:\n{tb}", flush=True)
                result = None
            my_results.append(result)

            # 每个进程单独输出处理进度
            percent = (i + 1) / nblocks * 100
            bar_len = 30
            filled_len = int(bar_len * (i + 1) / nblocks)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"[Rank {rank}] Processing: [{bar}] {i + 1}/{nblocks} ({percent:.1f}%)", flush=True)

        # 非主进程发送结果回 rank 0
        if rank != 0:
            comm.send(my_results, dest=0, tag=rank)
        else:
            # 主进程先收自己的结果
            final_results = my_results.copy()
            total_expected = size
            collected = 1  # 主进程自己已经处理

            print_progress(collected, total_expected)
            # 然后接收其他进程的结果
            for i in range(1, size):
                worker_results = comm.recv(source=i, tag=i)
                final_results.extend(worker_results)
                collected += 1
                print_progress(collected, total_expected)
            
            # 打印最终结果
            print("All results collected:")
            for res in final_results:
                print(res)
    def save_to_HDF5(self,output_file,standard_blocks):
        with h5py.File(output_file, 'w') as f:
            for i, block in enumerate(standard_blocks):
                group = f.create_group(f"block_{i}")
                for key, value in block.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, dict):
                        subgroup = group.create_group(key)
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                subgroup.create_dataset(k, data=v, compression='gzip')
                            else:
                                subgroup.attrs[k] = v
                    else:
                        group.attrs[key] = value

    def convert_to_standard_format(self,blocks_to_process):
        standard_blocks = []
        for block in blocks_to_process:
            block_dict = {
                'judaca': getattr(block, 'judaca', False),
                'Mf_A1s': np.array(block.Mf_A1s) if isinstance(getattr(block, 'Mf_A1s', None), np.ndarray) else getattr(block, 'Mf_A1s', None),
                'Mf_A2s': np.array(block.Mf_A2s) if isinstance(getattr(block, 'Mf_A2s', None), np.ndarray) else getattr(block, 'Mf_A2s', None),
                'Mf_Bs': np.array(block.Mf_Bs) if isinstance(getattr(block, 'Mf_Bs', None), np.ndarray) else getattr(block, 'Mf_Bs', None),
                'Mf_A1d': np.array(block.Mf_A1d) if isinstance(getattr(block, 'Mf_A1d', None), np.ndarray) else getattr(block, 'Mf_A1d', None),
                'Mf_A2d': np.array(block.Mf_A2d) if isinstance(getattr(block, 'Mf_A2d', None), np.ndarray) else getattr(block, 'Mf_A2d', None),
                'Mf_Bd': np.array(block.Mf_Bd) if isinstance(getattr(block, 'Mf_Bd', None), np.ndarray) else getattr(block, 'Mf_Bd', None),
                'ACA_dictS': {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in getattr(block, 'ACA_dictS', {}).items()},
                'ACA_dictD': {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in getattr(block, 'ACA_dictD', {}).items()},
                'row_cluster': np.array(block.row_cluster) if isinstance(getattr(block, 'row_cluster', None), np.ndarray) else getattr(block, 'row_cluster', None),
                'col_cluster': np.array(block.col_cluster) if isinstance(getattr(block, 'col_cluster', None), np.ndarray) else getattr(block, 'col_cluster', None),
            }
            standard_blocks.append(block_dict)
        return standard_blocks
    #Assign tasks in each submatrix for calculating green functions
    def master(self,dir,blocks_to_process,num_workers,save_corefunc=False):
        task_id = 1
        next_task = 0
        active_workers = num_workers
        
        #print('label',len(self.Label))
        if(len(blocks_to_process)==0):
            self.collect_blocks(self.root_block)
            #self.blocks_to_process.sort(key=lambda block: len(block.row_cluster), reverse=True)
        else:
            self.blocks_to_process=blocks_to_process
        self.Label=np.zeros(len(self.blocks_to_process))

        tasks_total = len(self.blocks_to_process)  # task count
        k=0
        for i in range(tasks_total):
            block=self.blocks_to_process[i]
            
            if(len(block.row_cluster)>500 or len(block.col_cluster)>500):
                k=k+1
                #print(len(block.row_cluster))
            
        print('tasks_total',k,tasks_total)

        # **Initial Task Assignment**
        for worker_rank in range(1, num_workers + 1):
            MPI.COMM_WORLD.send({'task':self.blocks_to_process[task_id-1],'task_id':task_id}, dest=worker_rank, tag=TASK_TAG)
            print(f"Master: assign task {task_id} to Worker {worker_rank}, size: {len(self.blocks_to_process[task_id-1].row_cluster),len(self.blocks_to_process[task_id-1].col_cluster)}", flush=True)
            task_id += 1
        
        finish_task=0
        pending_tasks = {} 
        # **Loop and wait for the worker to return the result and assign new tasks**
        while active_workers > 0:
            status = MPI.Status()
            if MPI.COMM_WORLD.Iprobe(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status):
                result = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG)
                

                #print(result)
                jud_already_calsvd=result['jud_already_calsvd']
                worker_rank = result['worker']
                block=result['result']
                #print(block.Mf_A1s)
                rece_id=result['task_id']
                print(f"Worker {MPI.COMM_WORLD.Get_rank()} receive the task result {rece_id}, size = {len(block.row_cluster),len(block.col_cluster)}", flush=True)
                
                if(jud_already_calsvd==True):
                    print(f"Master: Worker {worker_rank} for task {result['task']} already done before, size = {len(result['result'].row_cluster),len(result['result'].col_cluster)}", flush=True)
                    finish_task=finish_task+1
                else:
                    #worker_rank = result['worker']
                    self.blocks_to_process[rece_id-1]=block
                    #print(len(Label))
                    self.Label[rece_id-1]=1
                    #if(rece_id-1<len(self.blocks_to_process)):
                    #    if(len(self.blocks_to_process[rece_id-1].row_cluster)>300):
                    #        dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
                    finish_task=finish_task+1
                    print(f"Master: Worker {worker_rank} finish task {result['task_id']} , size = {len(result['result'].row_cluster),len(result['result'].col_cluster)}", flush=True)
                    #print(f"{finish_task}/ {tasks_total}", flush=True)
                print(f"Master: complete {finish_task}/ {tasks_total}", flush=True)
                # **If there are still tasks, continue to assign**
                if task_id <= tasks_total:
                    # try:
                    #     pickle.dumps(self.blocks_to_process[task_id-1])
                    # except Exception as e:
                    #     print(f"Cannot pickle block {task_id-1}: {e}")
                    #     raise
                    MPI.COMM_WORLD.send({'task':self.blocks_to_process[task_id-1],'task_id':task_id}, dest=worker_rank, tag=TASK_TAG)
                    print(f"Master: assign task {task_id} to Worker {worker_rank}, size = {len(self.blocks_to_process[task_id-1].row_cluster),len(self.blocks_to_process[task_id-1].col_cluster)}", flush=True)
                    #time.sleep(5)
                    pending_tasks[task_id] = (worker_rank, time.time())

                    
                    task_id += 1
                # else:
                #     active_workers -= 1
                #     print(f"active_workers remain {active_workers}", flush=True)
                #     # **No more tasks, send end signal**
                #     MPI.COMM_WORLD.send(None, dest=worker_rank, tag=STOP_TAG)
                if(finish_task==tasks_total):
                    for worker_rank in range(1, num_workers + 1):
                        active_workers -= 1
                        print(f"active_workers remain {active_workers}", flush=True)
                        MPI.COMM_WORLD.send(None, dest=worker_rank, tag=STOP_TAG)   

        if(save_corefunc==True):
            dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
            #blocks_to_process_trans=self.convert_to_standard_format(self.blocks_to_process)
            #self.save_to_HDF5(dir+ "/blocks_to_process.hdf5",blocks_to_process_trans)
        print("Master: all tasks completed.", flush=True)
    
    

    #Calculate ACA
    def worker(self):
        while True:
            print(f"[Worker {MPI.COMM_WORLD.Get_rank()}] waiting for task...", flush=True)
            status = MPI.Status()  
            rectask = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)

            # Check whether it is a termination signal through status.tag
            if status.tag == STOP_TAG:
                print(f"Worker {MPI.COMM_WORLD.Get_rank()}: end receive message, exit.", flush=True)
                #MPI.COMM_WORLD.Abort(1)
                break
            
            jud_already_calsvd=False
            # **执行任务**
            if(status.tag == TASK_TAG):
                #task_comf=0
                #print('task',task, flush=True)
                block=rectask['task']
                block=self.ACA_worker(block)
                # if hasattr(block, 'judproc') and block.judproc:
                #     jud_already_calsvd=True
                #     print(f"Worker {MPI.COMM_WORLD.Get_rank()}: skipping SVD(or ACA) for task {rectask['task_id']} (judproc=True)", flush=True)
                    
                # else:
                #     try:
                #         #block=self.svd_worker(block)
                #         block=self.ACA_worker(block)
                #     except:
                #         print(f"Worker: svd(ACA) failed for worker {rectask['task_id']}", flush=True)
                #     #    print(f"Worker: svd(ACA) of task {rectask['task_id']} failed for worker {MPI.COMM_WORLD.Get_rank()} ", flush=True)
                #     #    break
                
                #print(block.Mf_A1s)
                #result = task * task  # 假设任务是计算平方
                print(f"Worker {MPI.COMM_WORLD.Get_rank()} send the the task result {rectask['task_id']}, size = {len(block.row_cluster),len(block.col_cluster)}", flush=True)

                # **返回结果**
                #task_comf=1
                #print(f"Worker {MPI.COMM_WORLD.Get_rank()}: send block result, exit.", flush=True)
                MPI.COMM_WORLD.send({'worker': MPI.COMM_WORLD.Get_rank(), 'task_id': rectask['task_id'], 'result': block,'jud_already_calsvd':jud_already_calsvd}, dest=0, tag=RESULT_TAG)
                

    def calc_Hmatrix_ACA(self,dir,save_corefunc=False):
        blocks_to_process=[]
        self.collect_blocks(self.root_block)
        for i in range(len(self.blocks_to_process)):
            print('blocks_to_process:%d/%d'%(i,len(self.blocks_to_process)))
            result=self.ACA_worker(self.blocks_to_process[i])
            blocks_to_process.append(result)
        
        if(save_corefunc==True):
            dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
            #blocks_to_process_trans=self.convert_to_standard_format(self.blocks_to_process)
            #self.save_to_HDF5(dir+ "/blocks_to_process.hdf5",blocks_to_process_trans)
        print("Master: all blocks_to_process completed.", flush=True)
        return blocks_to_process

    def collect_blocks(self, block):
        """ Recursively traverse all leaf nodes and collect blocks that need to calculate ACA """
        if block.is_leaf():
            self.blocks_to_process.append(block)
        else:
            for child in block.children:
                self.collect_blocks(child)


    def svd_worker(self, block):
        """Calculate the SVD of a single block """
        if(len(block.row_cluster) > self.thread_svd and len(block.col_cluster) > self.thread_svd):
            block.judsvd=True
            #if(rank==0):
            #print('rank:',rank,'  Progress of stress function calculation:',self.progress_tra,' / ',self.size_local_blocks,' size:',len(block.row_cluster),len(block.col_cluster), flush=True)
            A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(block.row_cluster,block.col_cluster)
            self.progress_tra=self.progress_tra+1.0
            k=max(int(len(block.row_cluster)/20),self.mini_leaf)
            k1=min(len(block.row_cluster),len(block.col_cluster))
            if(k>=k1):
                k=k1-2
            #A1s = np.random.rand(9000, 4200)
            #print('!!!!!!!!!!!!!!!!!!!!!',A1s.shape)
            #U, S, Vt = np.linalg.svd(A1s, full_matrices=False)
            U, S, Vt = svds(A1s,k=k)
            block.U_1s=U
            block.S_1s=S
            block.Vt_1s=Vt
            

            U, S, Vt = svds(A2s, k=k)
            block.U_2s=U
            block.S_2s=S
            block.Vt_2s=Vt

            if not np.all(Bs == 0):
                U, S, Vt = svds(Bs, k=k)
                block.U_Bs=U
                block.S_Bs=S
                block.Vt_Bs=Vt

            U, S, Vt = svds(A1d, k=k)
            block.U_1d=U
            block.S_1d=S
            block.Vt_1d=Vt

            U, S, Vt = svds(A2d, k=k)
            block.U_2d=U
            block.S_2d=S
            block.Vt_2d=Vt

            if not np.all(Bd == 0):
                U, S, Vt = svds(Bd, k=k)
                block.U_Bd=U
                block.S_Bd=S
                block.Vt_Bd=Vt
                
            #else:
            #    print(f"Bd zero !!!! ", flush=True)
            
            if not np.all(Bd == 0):
                self.size += (U.nbytes + S.nbytes + Vt.nbytes)*6.0/(1024*1024)
            else:
                self.size += (U.nbytes + S.nbytes + Vt.nbytes)*4.0/(1024*1024)
            
        else:
            #if(rank==0):
            #print('rank:',rank,'  Progress of stress function calculation:',rank,self.progress_tra/self.size_local_blocks, flush=True)
            #print('rank:',rank,'  Progress of stress function calculation:',self.progress_tra,' / ',self.size_local_blocks,' size:',len(block.row_cluster),len(block.col_cluster), flush=True)
            A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_fromC(block.row_cluster,block.col_cluster)
            self.progress_tra=self.progress_tra+1.0
            block.Mf_A1s=A1s
            block.Mf_A2s=A2s
            block.Mf_Bs=Bs
            block.Mf_A1d=A1d
            block.Mf_A2d=A2d
            block.Mf_Bd=Bd
            self.size +=A1s.nbytes*6/(1024*1024)
            
        block.judproc=True
        return block
    
    def parallel_cells_scatter_send(self):
        N=len(self.eleVec)
        index0=np.arange(0,N,1)
        local_index = None
        if rank == 0:
            print('Assign cells for Dilantacy calculation:', N)
    
            # Manually distribute tasks evenly
            counts = [N // size] * size
            for i in range(N % size):
                counts[i] += 1
            task_chunks = []
            start = 0
            for c in counts:
                task_chunks.append(index0[start:start+c])
                start += c
            for i in range(1, size):
                comm.send(task_chunks[i], dest=i, tag=77)
            local_index = task_chunks[0] 
        else:
            #Non-zero process receiving tasks
            
            local_index = comm.recv(source=0, tag=77)
        print('rank',rank,' cells for P calculation',len(local_index))
        return local_index
    
    #Assign missions for forward iteration each rank with completed blocks submatrice
    def parallel_block_scatter_send(self, blocks_to_process, plotHmatrix=False):
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # size = comm.Get_size()

        local_blocks = None
        n_task_per_proc=50
        if rank == 0:
            num_blocks = len(blocks_to_process)
            print('num_blocks:', num_blocks)
    
            # Manually distribute tasks evenly
            counts = [num_blocks // size] * size
            for i in range(num_blocks % size):
                counts[i] += 1
    
            task_chunks = []
            start = 0
            for c in counts:
                task_chunks.append(blocks_to_process[start:start+c])
                start += c
    
            #print('task_chunks length per process:', [len(c) for c in task_chunks])
    
            # Send to other processes
            for i in range(1, size):
                
                batch=int(len(task_chunks[i])/n_task_per_proc)
                start = 0
                #Send in sequence to reduce the load of each transmission
                for j in range(n_task_per_proc):
                    #task_dict=self.trans_class_to_dict(task_chunks[i])
                    if(j<n_task_per_proc-1):
                        comm.send(task_chunks[i][start:start+batch], dest=i, tag=77+j)
                    else:
                        comm.send(task_chunks[i][start:], dest=i, tag=77+j)
                    start += batch
                    #comm.send(task_chunks[i], dest=i, tag=77)
            local_blocks = task_chunks[0]  # rank=0 his task
            #print('rank ',rank,len(local_blocks))
    
        else:
            # Non-zero process receiving tasks
            #print('rank',rank,len(task_chunks[i]))
            #local_blocks = comm.recv(source=0, tag=77)
            #print('rank',rank,len(local_blocks))

            local_blocks = []
            for j in range(n_task_per_proc):
                task = comm.recv(source=0, tag=77 + j)
                for k in range(len(task)):
                    local_blocks.append(task[k])
        print('rank ',rank,'Hmatirx sub-blocks number',len(local_blocks))
        # Optional drawing processing
        if plotHmatrix:
            # Receive data from each process step by step
            #if(rank==0):
            # gathered_blocks = []
            # for i in range(size):
            #     if rank == i:
            #         gathered_blocks.append(local_blocks)
            #     comm.barrier()  
            if rank == 0:
                #print('gathered_blocks:',len(gathered_blocks))
                self.blocks_plot_mpi(task_chunks)
    
        return local_blocks
    

    '''Matrix and vector product calculation'''
    def blocks_process_MVM(self,xvector,blocks_to_process,type):
        yvector=np.zeros(len(xvector))
        #start_time = time.time()  
        #for k in range(100):
        #A1s=np.load('WMF_core/A1s.npy')
        if(type=='A1s'):
            for i in range(len(blocks_to_process)):
                #print(xvector.shape,np.max(blocks_to_process[i].col_cluster))
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_1s @ (blocks_to_process[i].S_1s * (blocks_to_process[i].Vt_1s @ x_))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    Ax_rsvd = blocks_to_process[i].ACA_dictS['U_ACA_A1s'].dot(blocks_to_process[i].ACA_dictS['V_ACA_A1s'].dot(x_))
                    #print(Ax_rsvd)
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A1s @ x_
                
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd
        if(type=='A2s'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_2s @ (blocks_to_process[i].S_2s * (blocks_to_process[i].Vt_2s @ x_))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    Ax_rsvd = blocks_to_process[i].ACA_dictS['U_ACA_A2s'].dot(blocks_to_process[i].ACA_dictS['V_ACA_A2s'].dot(x_))
                    
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A2s @ x_
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd

        if(type=='Bs'):
            for i in range(len(blocks_to_process)):
                
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    if(len(blocks_to_process[i].Vt_Bs)>0):
                        Ax_rsvd = blocks_to_process[i].U_Bs @ (blocks_to_process[i].S_Bs * (blocks_to_process[i].Vt_Bs @ x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].row_cluster))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    if(len(blocks_to_process[i].ACA_dictS['U_ACA_Bs'])>0):
                        Ax_rsvd = blocks_to_process[i].ACA_dictS['U_ACA_Bs'].dot(blocks_to_process[i].ACA_dictS['V_ACA_Bs'].dot(x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].row_cluster))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_Bs @ x_
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd

        if(type=='A1d'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_1d @ (blocks_to_process[i].S_1d * (blocks_to_process[i].Vt_1d @ x_))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    Ax_rsvd = blocks_to_process[i].ACA_dictD['U_ACA_A1d'].dot(blocks_to_process[i].ACA_dictD['V_ACA_A1d'].dot(x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A1d @ x_
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd

        if(type=='A2d'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_2d @ (blocks_to_process[i].S_2d * (blocks_to_process[i].Vt_2d @ x_))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    Ax_rsvd = blocks_to_process[i].ACA_dictD['U_ACA_A2d'].dot(blocks_to_process[i].ACA_dictD['V_ACA_A2d'].dot(x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A2d @ x_
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd

        if(type=='Bd'):
            for i in range(len(blocks_to_process)):
                
                x_=xvector[blocks_to_process[i].col_cluster]
                if(blocks_to_process[i].judsvd==True):
                    if(len(blocks_to_process[i].Vt_Bd)>0):
                        Ax_rsvd = blocks_to_process[i].U_Bd @ (blocks_to_process[i].S_Bd * (blocks_to_process[i].Vt_Bd @ x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].row_cluster))
                elif(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                    if(len(blocks_to_process[i].ACA_dictD['U_ACA_Bd'])>0):
                        Ax_rsvd = blocks_to_process[i].ACA_dictD['U_ACA_Bd'].dot(blocks_to_process[i].ACA_dictD['V_ACA_Bd'].dot(x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].row_cluster))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_Bd @ x_
                yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd
        return yvector
    

    def blocks_process_MVM_gpu(self, xvector, blocks_to_process, type):
        import cupy as cp
        # Convert input vector to CuPy array
        xvector = cp.array(xvector)
        # Initialize output vector on GPU
        yvector = cp.zeros(len(xvector))
        
        if type == 'A1s':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    # SVD-based low-rank: y = U @ (S * (Vt @ x))
                    U = cp.array(blocks_to_process[i].U_1s)
                    S = cp.array(blocks_to_process[i].S_1s)
                    Vt = cp.array(blocks_to_process[i].Vt_1s)
                    Ax_rsvd = U @ (S * (Vt @ x_))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    # ACA-based low-rank: y = U_ACA @ (V_ACA @ x)
                    U_ACA = cp.array(blocks_to_process[i].ACA_dictS['U_ACA_A1s'])
                    V_ACA = cp.array(blocks_to_process[i].ACA_dictS['V_ACA_A1s'])
                    Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                else:
                    # Dense matrix multiplication
                    Mf = cp.array(blocks_to_process[i].Mf_A1s)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        elif type == 'A2s':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    U = cp.array(blocks_to_process[i].U_2s)
                    S = cp.array(blocks_to_process[i].S_2s)
                    Vt = cp.array(blocks_to_process[i].Vt_2s)
                    Ax_rsvd = U @ (S * (Vt @ x_))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    U_ACA = cp.array(blocks_to_process[i].ACA_dictS['U_ACA_A2s'])
                    V_ACA = cp.array(blocks_to_process[i].ACA_dictS['V_ACA_A2s'])
                    Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                else:
                    Mf = cp.array(blocks_to_process[i].Mf_A2s)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        elif type == 'Bs':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    if len(blocks_to_process[i].Vt_Bs) > 0:
                        U = cp.array(blocks_to_process[i].U_Bs)
                        S = cp.array(blocks_to_process[i].S_Bs)
                        Vt = cp.array(blocks_to_process[i].Vt_Bs)
                        Ax_rsvd = U @ (S * (Vt @ x_))
                    else:
                        Ax_rsvd = cp.zeros(len(blocks_to_process[i].row_cluster))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    if len(blocks_to_process[i].ACA_dictS['U_ACA_Bs']) > 0:
                        U_ACA = cp.array(blocks_to_process[i].ACA_dictS['U_ACA_Bs'])
                        V_ACA = cp.array(blocks_to_process[i].ACA_dictS['V_ACA_Bs'])
                        Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                    else:
                        Ax_rsvd = cp.zeros(len(blocks_to_process[i].row_cluster))
                else:
                    Mf = cp.array(blocks_to_process[i].Mf_Bs)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        elif type == 'A1d':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    U = cp.array(blocks_to_process[i].U_1d)
                    S = cp.array(blocks_to_process[i].S_1d)
                    Vt = cp.array(blocks_to_process[i].Vt_1d)
                    Ax_rsvd = U @ (S * (Vt @ x_))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    U_ACA = cp.array(blocks_to_process[i].ACA_dictD['U_ACA_A1d'])
                    V_ACA = cp.array(blocks_to_process[i].ACA_dictD['V_ACA_A1d'])
                    Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                else:
                    Mf = cp.array(blocks_to_process[i].Mf_A1d)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        elif type == 'A2d':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    U = cp.array(blocks_to_process[i].U_2d)
                    S = cp.array(blocks_to_process[i].S_2d)
                    Vt = cp.array(blocks_to_process[i].Vt_2d)
                    Ax_rsvd = U @ (S * (Vt @ x_))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    U_ACA = cp.array(blocks_to_process[i].ACA_dictD['U_ACA_A2d'])
                    V_ACA = cp.array(blocks_to_process[i].ACA_dictD['V_ACA_A2d'])
                    Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                else:
                    Mf = cp.array(blocks_to_process[i].Mf_A2d)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        elif type == 'Bd':
            for i in range(len(blocks_to_process)):
                x_ = xvector[blocks_to_process[i].col_cluster]
                if blocks_to_process[i].judsvd:
                    if len(blocks_to_process[i].Vt_Bd) > 0:
                        U = cp.array(blocks_to_process[i].U_Bd)
                        S = cp.array(blocks_to_process[i].S_Bd)
                        Vt = cp.array(blocks_to_process[i].Vt_Bd)
                        Ax_rsvd = U @ (S * (Vt @ x_))
                    else:
                        Ax_rsvd = cp.zeros(len(blocks_to_process[i].row_cluster))
                elif hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca:
                    if len(blocks_to_process[i].ACA_dictD['U_ACA_Bd']) > 0:
                        U_ACA = cp.array(blocks_to_process[i].ACA_dictD['U_ACA_Bd'])
                        V_ACA = cp.array(blocks_to_process[i].ACA_dictD['V_ACA_Bd'])
                        Ax_rsvd = U_ACA.dot(V_ACA.dot(x_))
                    else:
                        Ax_rsvd = cp.zeros(len(blocks_to_process[i].row_cluster))
                else:
                    Mf = cp.array(blocks_to_process[i].Mf_Bd)
                    Ax_rsvd = Mf @ x_
                yvector[blocks_to_process[i].row_cluster] += Ax_rsvd

        # Convert result back to NumPy array for compatibility
        return cp.asnumpy(yvector)

    #Hmatrix structure drawing
    def blocks_plot_mpi(self,gathered_results):
        color1=['darkred','darkblue','lime','blue','y','cyan','darkgreen','steelblue','tomato','chocolate','slateblue']*size
        plt.figure(figsize=(10,10))
        #plt.rcParams['font.family'] = 'Arial'
        plt.rcParams.update({'font.size': 14})
      

        #print('gathered_results',len(gathered_results))
        for i in range(len(gathered_results)):
        #for i in range(4):
            for j in range(len(gathered_results[i])):
                block=gathered_results[i][j]
                rowleftsize=len(block.row_cluster)
                colleftsize=len(block.col_cluster)
                
                midr=block.row_index[0]+rowleftsize
                midc=block.col_index[0]+colleftsize
                plt.plot([block.col_index[0],block.col_index[-1]],[midr,midr],c=color1[i])
                plt.plot([midc,midc],[block.row_index[0],block.row_index[-1]],c=color1[i])
        
        plt.xlim(0,len(self.xg)-1)
        plt.ylim(0,len(self.xg)-1)
        plt.savefig('HmatrixStru_mpi.eps',dpi=500)
        #plt.show()




    def apply_spr_values(self,sparse_csr):
        rows, cols = sparse_csr.nonzero()
        print('Store leaf stress data index in sparse matrix ...')
        #self.A1s_spr,self.A2s_spr,self.Bs_spr,self.A1d_spr,self.A2d_spr,self.Bd_spr=self.calc_stressfunc(rows, cols)
        
        # for i in range(sparse_csr.shape[0]):
        #     index1=np.where(rows==i)[0]
        #     tar_cols=cols[index1]

        #     #clac stress function
        #     self.A1s_spr,self.A2s_spr,self.Bs_spr,self.A1d_spr,self.A2d_spr,self.Bd_spr=self.calc_stressfunc([i], tar_cols)
        #     values=A1[i,tar_cols]
        #     sparse_csr[i,tar_cols]=values
        print('sparse matrix store completed')


    def traverse_and_apply(self, block=None, depth=0, max_level=2):
        """
        Recursively traverse BlockTree and perform low-rank approximation when conditions are met.
        :param block: the block currently traversed
        :param depth: the current level
        :param max_level: the maximum level at which low-rank approximation is applied
        """
        if block is None:
            block = self.root_block

        indent = "    " * depth  
        if block.is_leaf():
            print(f"{indent}- Leaf Block: Level {block.level}, Rows {block.row_cluster}, Cols {block.col_cluster}, Data Shape: {block.data.shape}")
            if block.level <= max_level:  # Only make low-rank approximation for leaf nodes below the max_level layer
                block.apply_low_rank_approximation()
                print(f"{indent}  (Applied Low-Rank Approximation)")
        else:
            print(f"{indent}- Composite Block: Level {block.level}, Rows {block.row_cluster}, Cols {block.col_cluster}, {len(block.children)} sub-blocks")
            for child in block.children:
                self.traverse_and_apply(child, depth + 1, max_level)


# --------------------------
# 递归构造 BlockTree
# --------------------------

def create_recursive_blocks(row_cluster,col_cluster,row_index,col_index,points,plotHmatrix,mini_leaf=16, depth=0):
    """
    Recursively create BlockTree
    :param matrix: target matrix
    :param row_range: current block row index range
    :param col_range: current block column index range
    :param max_depth: maximum recursive depth
    :param depth: current depth
    :return: Block object
    """

    
    
    jud_admis=is_admissible(row_cluster.indices,col_cluster.indices,points, eta=2.0)
    #if(len(row_cluster.indices)>5000 or len(col_cluster.indices)>5000):
    #    jud_admis=False
    if jud_admis == True or (len(row_cluster.indices) <= mini_leaf or len(col_cluster.indices) <= mini_leaf):
        # Termination condition: reaching the maximum depth or the matrix block is too small to be further divided
        return Block(row_cluster.indices,col_cluster.indices,row_index,col_index, level=depth)

    row_cluster_left=row_cluster.left
    row_cluster_right=row_cluster.right
    col_cluster_left=col_cluster.left
    col_cluster_right=col_cluster.right

    if((row_cluster_left==None) or (row_cluster_right==None) or (col_cluster_left==None) or (col_cluster_right==None)):
        return Block(row_cluster.indices,col_cluster.indices,row_index,col_index, level=depth)

    rowleftsize=len(row_cluster_left.indices)
    #rowrightsize=len(row_cluster_right.indices)
    colleftsize=len(col_cluster_left.indices)
    #colrightsize=len(row_cluster_left.indices)
    #print('depth',depth)
    if(plotHmatrix==True):
        #print('!!!!!!!!!!!!!!')
        # midr=(row_index[0]+row_index[-1])/2.0
        # midc=(col_index[0]+col_index[-1])/2.0
        midr=row_index[0]+rowleftsize
        midc=col_index[0]+colleftsize
        plt.plot([col_index[0],col_index[-1]],[midr,midr],c='red')
        plt.plot([midc,midc],[row_index[0],row_index[-1]],c='red')
        #plt.show()



    

    

    # 创建四个子块
    children = [
        create_recursive_blocks(row_cluster_left, col_cluster_left,
                        row_index[:rowleftsize],col_index[:colleftsize],points, plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_left, col_cluster_right,
                        row_index[:rowleftsize],col_index[colleftsize:],points, plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_right, col_cluster_left, 
                        row_index[rowleftsize:],col_index[:colleftsize],points,plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_right, col_cluster_right, 
                        row_index[rowleftsize:],col_index[colleftsize:],points,plotHmatrix,mini_leaf,depth + 1),
    ]

    return Block(row_cluster.indices,col_cluster.indices,row_index,col_index,children=children, level=depth)



def createHmatrix(xg,nodelst,elelst,eleVec,mu_,lambda_,halfspace_jud,mini_leaf=32,plotHmatrix=False,GPU=False):
    
    if(GPU==True):
        global cp
        import cupy as cp
    
    
    #useC=comm.bcast(useC, root=0)

    cluster = np.arange(len(xg))
    cluster_raw =build_block_tree(cluster, xg)
    cluster_col =build_block_tree(cluster, xg)

    #print(len(cluster_raw.left.indices))
    #testblock.print_tree(tree_root)
    if(plotHmatrix==True):
        plt.figure(figsize=(10,10))
    print('start Recursively traverse create the BlockTree.')
    root_block = create_recursive_blocks(cluster_raw, cluster_col,cluster,cluster,xg,plotHmatrix,mini_leaf)
    print('Recursively traverse create the BlockTree.')
    tree_block = BlockTree(root_block,nodelst,elelst,eleVec,mu_,lambda_, xg,halfspace_jud,mini_leaf)
    print('Recursively traverse create the BlockTree completed.')
    #print('Recursively traverse the BlockTree to obtain the interpolation index positions.')
    #tree_block.traverse()
    #print('Recursively traverse completed.')
    #tree_block.Intep_rowindex=np.concatenate(tree_block.Intep_rowindex)
    #tree_block.Intep_colindex=np.concatenate(tree_block.Intep_colindex)
    #print(root_block.children[0].children[3].col_index)

    #sparse_matrix_coo = coo_matrix((np.ones(len(tree_block.Intep_rowindex)), 
    #        (tree_block.Intep_rowindex, tree_block.Intep_colindex)), shape=(len(xg), len(xg)))
    #sparse_csr = sparse_matrix_coo.tocsr()
    #sparse_csr=calc_Sgreenfunc_spr(sparse_csr)
    #tree_block.sparse_csr=sparse_csr
    #tree_block.apply_spr_values(sparse_csr)

    #print(root_block.row_cluster)
    # print('Recursively traverse the BlockTree to apply SVD...')
    # tree_block.traverse_SVD()
    # print('Apply SVD complete.')

    if(plotHmatrix==True):
        plt.xlim(cluster[0],cluster[-1])
        plt.ylim(cluster[0],cluster[-1])
        plt.savefig('HmatrixStru.png',dpi=500)
        #plt.show()
    return tree_block





