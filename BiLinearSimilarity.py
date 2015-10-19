# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.utils.extmath import randomized_svd
from scipy.optimize import minimize

nprs = np.random.RandomState(4217)
random.seed(4217)


#####################################################################
#####################   RDML    
# Jin, R., Wang, S., & Zhou, Y. (2009). Regularized distance metric learning: Theory and algorithm. In Advances in neural information processing systems (pp. 862-870).

def RDML(X,Y,lmbda=0.1,T=1000):
    # X     - nxd input matrix with 
    #           n - number of patterns
    #           d - number of features
    # Y     - n-dimensional array of labels
    # lmbda - learning rate 
    # T     - max number of iterations 

    if not isinstance(X, np.matrix):
        X=np.matrix(X)

    #get dimensions
    n,d = X.shape
    #initialize A
    A=np.matrix(np.zeros((d,d)))
    for i in xrange(T):
        #chose a pair. Use random sample to chose a pair without replacement.
        pair=np.array(random.sample(xrange(n),2))
        #get labels
        #import IPython
        #IPython.embed()
        ys=Y[pair]

        yt=-1.0
        if ys[0]==ys[1]: yt=1.0
        #get random rows
        xs=X[pair,:]
        #classify
        xd=xs[0,:]-xs[1,:]
        if yt*(xd*A*xd.T)>0.0:
            #correctly classified, don't adapt A
            continue
        #use near psd projection
        #A=nearPSD(A-lmbda*yt*xd.T*xd,epsilon=10**-10)
        #use approximate solution derived in RDML paper
        if yt==-1:
            A=A-lmbda*yt*xd.T*xd
        else:
            lmbda_t=lambda_CG(A,xd,lmbda)
            A=A-lmbda_t*yt*xd.T*xd
    return A


def lambda_CG(A,xd,lmbda):
    # using conjugate gradient method
    # Shewchuk, J. R. (1994). An introduction to the conjugate gradient method without the agonizing pain.
    result = minimize(f_loss,x0=np.zeros(xd.shape[1]),hess=f_hess,options={'disp':False},method='Newton-CG',jac=f_grad,args=(A,xd.T))
    if result.fun==0.0:
        return 0.0
    else:
        return max(0,min(lmbda,(-result.fun)**-1))

def f_loss(u,A,xdT):
    u=np.matrix(u)
    return np.squeeze(np.asarray(-2.0*u*xdT+u*A*u.T))
    
def f_grad(u,A,xdT):
    u=np.matrix(u)
    return np.squeeze(np.asarray(-2.0*xdT+(A+A.T)*u.T))

def f_hess(u,A,xdT):
    return np.asarray(A+A.T)

#####################################################################
#### LowRankBiLinear #####
# Method due to Liu et al. (2015) Low-Rank Similarity Metric Learning in High Dimensions

#proximal mapping
def Tfunc(M,theta):
   n,m = M.shape
   out = np.zeros((n,m))
   for i in xrange(n):
     for j in xrange(m):
        if M[i,j]>theta: out[i,j]=M[i,j]-theta
        elif M[i,j]<0.0: out[i,j]=M[i,j]
   return(out)

#faster proximal mapping
def Tfunc_fast(M,theta):
    A=M.copy() 
    A[M>0]=0
    return np.maximum(M-theta,0)+A

#projection onto the PSD cone
def nearPSD(A,epsilon=0.0):
   n = A.shape[0]
   if n==1: return(np.maximum(A,epsilon))
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1.0/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def LowRankBiLinear(m,X,Y,alpha,eps,rho,tau,T,tol=1e-6,epsilon=10**-8):
    # m - dimension of similarity function
    # X - n x d data matrix. n << d. n observations, d dimensions
    # Y - n x 1 label vector
    # eps - margin parameter for the metric. eps >0, a small positive value
    # rho - rho >0, penalty parameter for the augmentation term of the lagrangian (method of multipliers)
    #       the dual variable update uses a step size equal to rho 
    # alpha - regularization strength
    # tau - step size for the W updates
    # T - iteration limit
    # epsilon parameter for the projection of a matrix onto the PSD cone

    n,d = X.shape
    #build pairwise label matrices Ym and Ytil
    #initialize Ym with ones
    Ym = -np.matrix(np.ones((n,n)))
    #init Ytil with eps
    Ytil = eps*np.matrix(np.ones((n,n)))
    for y in np.unique(Y):
      y_vec = Y==y
      Y_set=np.outer(y_vec,y_vec)
      Ym[Y_set]=1
      Ytil[Y_set]=1
    #transpose X
    X = X.T 
    #SVD projection
    U, E, V = randomized_svd(X,n_components=m,n_iter=5,random_state=17)
    U = np.asmatrix(U)
    E2 = np.matrix(np.diag(E**2))
    Xtil = U.T*X
    XtilT=Xtil.T
    I = np.matrix(np.identity(m))
    W = I.copy()
    L = np.matrix(np.zeros((n,n)))
    S = XtilT*Xtil
    #find W using linearized ADMM (alternating direction method of multipliers)
    for k in xrange(T):
      Z = Tfunc_fast(Ytil-np.multiply(Ym,S)-L,1.0/rho)
      G = alpha/rho*I+Xtil*np.multiply(Ym,Z-Ytil+L)*XtilT + E2*W*E2
      W = nearPSD(W-tau*G,epsilon)
      S = XtilT*W*Xtil
      Delta = Z-Ytil+np.multiply(Ym,S)
      L = L + Delta
      if np.sum(np.multiply(Delta,Delta))/n**2 <= tol:
        break
    E,H = np.linalg.eig(W)
    E = np.maximum(E,epsilon)
    out = U*H*np.diag(np.sqrt(E))
    return(out)#return the optimal low rank basis Lstar

#### LowRankBiLinear Example #####
#m = 2
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
#              [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#              [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
#              [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#              [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
#              [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, 0, 0, 0])
#eps = 0.01
#rho = 1.0
#alpha = 0.5
#tau = 0.01
#T = 100
#tol = 1e-6
#L = LowRankBiLinear(m,X,Y,eps,rho,tau,T)
#print L 
def test():
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import StratifiedKFold
    digits = load_digits()
    X=digits.data
    Y=digits.target
    # 10-fold cross validation
    skf = StratifiedKFold(Y, 10,random_state=17)
    for j,x in enumerate(skf):
        #RDML
        A=RDML(X[x[0]],Y[x[0]],lmbda=0.2,T=50*len(x[0]))
        clf = KNeighborsClassifier(n_neighbors=1,metric='pyfunc',method='brute',metric_params={"A": A},func=mahalanobis)

        #train
        clf.fit(X[x[0]],Y[x[0]])
        #predict
        print 'accuracy: '+str(clf.score(X[x[1]],Y[x[1]]))

    skf = StratifiedKFold(Y, 10,random_state=17)
    for j,x in enumerate(skf):
        #Liu et al
        L=LowRankBiLinear(40,X[x[0]],Y[x[0]],0.1,10**-6,0.1,0.1,100,tol=1e-6,epsilon=10**-8)
        A=L*L.T
        clf = KNeighborsClassifier(n_neighbors=1,metric='pyfunc',method='brute',metric_params={"A": A},func=distkernel)

        #train
        clf.fit(X[x[0]],Y[x[0]])
        #predict
        print 'accuracy: '+str(clf.score(X[x[1]],Y[x[1]]))
    


def mahalanobis(x,y,**kwargs):
    xd=np.matrix(x-y)
    try:
        return xd*kwargs["A"]*xd.T
    except:
        #we have to cheat since scikit seems to check the function but doesn't use the correct dimensions
        return np.linalg.norm(x-y)

def distkernel(x,y,**kwargs):
    try:
        return 1.0 - np.matrix(x)*kwargs["A"]*np.matrix(y.T)
    except:
        #we have to cheat since scikit seems to check the function but doesn't use the correct dimensions
        return np.linalg.norm(x-y)


def simfunc(x,y,**kwargs):
    return np.matrix(x)*kwargs["A"]*np.matrix(y.T)

if __name__ == "__main__":  
    test()
