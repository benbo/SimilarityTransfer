# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.extmath import randomized_svd

#####################################################################
#### LowRankBiLinear #####
# Method due to Liu et al. (2015) Low-Rank Similarity Metric Learning in High Dimensions

def Tfunc(M,theta):
   n,m = M.shape
   out = np.zeros((n,m))
   for i in xrange(n):
     for j in xrange(m):
        if M[i,j]>theta: out[i,j]=M[i,j]-theta
        elif M[i,j]<0.0: out[i,j]=M[i,j]
   return(out)

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   if n==1: return(np.maximum(A,epsilon))
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def LowRankBiLinear(m,X,Y,eps,rho,tau,T,tol=1e-6,epsilon=0.0):
    # m - dimension of similarity function
    # X - n x d data matrix. n << d. n observations, d dimensions
    # Y - n x 1 label vector
    # eps - ?
    # rho - ?
    # alpha - regularization strength
    # tau - ?
    # T - iteration limit
    n,d = X.shape
    Ym = -np.matrix(np.ones((n,n)))
    Ytil = eps*np.matrix(np.ones((n,n)))
    for y in np.unique(Y):
      y_vec = Y==y
      Ym = Ym + 2*np.outer(y_vec,y_vec)
      Ytil = Ytil + (1-eps)*np.outer(y_vec,y_vec)
    X = X.T 
    U, E, V = randomized_svd(X,n_components=m,n_iter=5,random_state=None)
    U = np.asmatrix(U)
    E2 = np.matrix(np.diag(E**2))
    Xtil = U.T*X
    W = np.matrix(np.identity(m))
    I = np.matrix(np.identity(m))
    L = np.matrix(np.zeros((n,n)))
    S = Xtil.T*Xtil
    for k in xrange(T):
      Z = Tfunc(Ytil-np.multiply(Ym,S)-L,1.0/rho)
      G = alpha/rho*I+Xtil*np.multiply(Ym,Z-Ytil+L)*Xtil.T + E2*W*E2
      W = nearPSD(W-tau*G,epsilon)
      S = Xtil.T*W*Xtil
      Delta = Z-Ytil+np.multiply(Ym,S)
      L = L + Delta
      if np.sum(np.multiply(Delta,Delta))/n**2 <= tol:
        break
    E,H = np.linalg.eig(W)
    E = np.maximum(E,epsilon)
    out = U*H*np.diag(np.sqrt(E))
    return(out)

#### LowRankBiLinear Example #####
#m = 2
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, 0, 0, 0])
#eps = 0.1
#rho = 1
#alpha = 0.5
#tau = 0.01
#T = 10
#tol = 1e-6
#L = LowRankBiLinear(m,X,Y,eps,rho,tau,T)

