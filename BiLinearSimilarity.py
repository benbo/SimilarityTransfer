# -*- coding: utf-8 -*-
import cmath
from numpy.polynomial.polynomial import polydiv
import numpy as np
#from bottleneck import argpartsort
import random
from sklearn.utils.extmath import randomized_svd
from scipy.optimize import minimize
from sklearn.utils.extmath import weighted_mode
from scipy.stats import mode
from itertools import izip


nprs = np.random.RandomState(4217)
random.seed(4217)

#####################   RDML  #####################   
# Jin, R., Wang, S., & Zhou, Y. (2009). 
#       Regularized distance metric learning: Theory and algorithm. 
#        In Advances in neural information processing systems (pp. 862-870).

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
    # using Newton CG to find approximate solution since gradient and hessian are easy to compute
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

#####################  LowRankBiLinear #####################
# Method due to Liu et al. (2015) 
#           Low-Rank Similarity Metric Learning in High Dimensions

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

#projection onto the PSD cone
def nearPSDsimple(A):
    n = A.shape[0]
    if n==1: return(np.maximum(A,epsilon))
    eigval, eigvec = np.linalg.eig(A)
    eig_ind=eigval>0
    B=np.zeros(A.shape)
    for val,vec in izip(eigval[eig_ind],eigvec[eig_ind]):
        B+=val*vec.T*vec
    return B

    



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
    if not isinstance(X, np.matrix):
        X=np.matrix(X)

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
    #U, E, _ = randomized_svd(X,n_components=m,n_iter=5,random_state=17)
    U, E, _ = np.linalg.svd(X, full_matrices=False)
    U=U[:,:m]
    E=E[:m]
    U = np.asmatrix(U)
    E2 = np.matrix(np.diag(E**2))
    Xtil = U.T*X
    XtilT=Xtil.T
    I = np.matrix(np.identity(m))
    W = I.copy()
    I =alpha/rho* I 
    L = np.matrix(np.zeros((n,n)))
    S = XtilT*Xtil
    #find W using linearized ADMM (alternating direction method of multipliers)
    for k in xrange(T):
      Z = Tfunc_fast(Ytil-np.multiply(Ym,S)-L,1.0/rho)
      G = I+Xtil*np.multiply(Ym,Z-Ytil+L)*XtilT + E2*W*E2
      W = nearPSDsimple(W-tau*G)
      S = XtilT*W*Xtil
      Delta = Z-Ytil+np.multiply(Ym,S)
      L = L + Delta
      if np.sum(np.multiply(Delta,Delta))/n**2 <= tol:
        break
    E,H = np.linalg.eig(W)
    E = np.maximum(E,epsilon)
    out = U*H*np.diag(np.sqrt(E))
    return(out)#return the optimal low rank basis Lstar

#####################################################################
#### OASIS_SIM #####
# Symmetric modification of OASIS by Kyle Miller

def OASIS_SIM(m,X,Y,C,itmax=10,batch_size=None,loss_tol=1e-3,epsilon=1e-10,Verbose=True,Lo=None):
 # m - projection dimension
 # X - data matrix, each row is an observation
 # Y - label vector (any type)
 # C - aggressiveness parameter, balances trade-off between fidelity to previous solution and correction for current loss
 # batch_size - number of random samples for which to calculate expected loss, default is the number of observations
 # loss_tol - stopping criterion, if estimate of expected loss falls below loss_tol we break
 # epsilon - size of zero. If absolute value of a number is less than epsilon, a number is considered zero
 if not isinstance(X, np.matrix):
   X=np.matrix(X)
 C_limit = C
 M,N = X.shape # M - number of observations, N - dimension of each
 if batch_size==None: batch_size = M
 Class = {}
 nC = {}
 for i in xrange(len(Y)):
   if Y[i] not in Class:
     Class[Y[i]] = []
     nC[Y[i]] = 0
   Class[Y[i]].append(i)
   nC[Y[i]] += 1
 nY = len(Y)
 if Verbose:
   print "Begining OASIS_SIM with "+str(M)+" observations,"
   print "  "+str(N)+" dimensions, and "+str(len(Class))+" classes."
 if Lo==None: L = np.asmatrix(np.ones((m,N)))/(m**2*N**2)
 else: L = Lo
 for k in xrange(itmax):
   totloss = 0.0
   totloss2 = 0.0
   max_root = 0.0
   min_root = 0.0
   max_root_denom = 0.0
   for i in xrange(batch_size):
     # randomly sample triplet
     idx = np.random.randint(0,nY)
     c = Y[idx]
     r_ref = X[idx,]
     pos_idx = Class[c][np.random.randint(0,nC[c])]
     neg_idx = np.random.randint(0,nY-nC[c])
     for neg_c in Class:
       if neg_c==c: continue
       if neg_idx>=nC[neg_c]: neg_idx -= nC[neg_c]
       else:
         neg_idx = Class[neg_c][neg_idx]
         break
     r_pos = X[pos_idx,]
     r_neg = X[neg_idx,]
     a = r_ref
     b = r_pos-r_neg
     ab = np.sum(a*b.T)
     aa = np.sum(a*a.T)
     bb = np.sum(b*b.T)
     if aa==0 or bb==0: continue
     La = L*a.T
     Lb = L*b.T
     aLLb = np.sum(La.T*Lb)
     aLLa = np.sum(La.T*La)
     bLLb = np.sum(Lb.T*Lb)
     if 1-aLLb<=0: continue # if non-positive loss, don't do anything
     totloss = totloss+(1-aLLb)
     totloss2 = totloss2+(1-aLLb)**2
     # f function coefficients
     A42 = 0.5*(aa*bb-ab**2)*(aa*bLLb+bb*aLLa-2*ab*aLLb)
     A32 = 2*(aa*bb-ab**2)*aLLb
     A22 = 0.5*(aa*bLLb+bb*aLLa+2*ab*aLLb)
     # q polynomial coefficients
     a4=(aa*bb-ab**2)**2
     a3=4.0*ab*(aa*bb-ab**2)
     a2=ab*(aLLa*bb+aa*bLLb)+((1.0-aLLb)-3.0)*(aa*bb-ab**2)+2.0*ab**2*((1.0-aLLb)+1.0)
     a1=-aLLa*bb+2.0*aLLb*ab-aa*bLLb-4.0*ab
     a0=1.0-aLLb
     #1. depress via substitution tau = y - a3/(4*a4) => y^4+ A*y^2+ B*y +C = 0
     A = (-3.0*a3**2/(8.0*a4)+a2)/a4
     B = (a3**3/(8.0*a4**2)-a2*a3/(2.0*a4)+a1)/a4
     C = (-3.0*a3**4/(256.0*a4**3)+a3**2*a2/(16.0*a4**2)-a3*a1/(4.0*a4)+a0)/a4
     #2. solve resolvant cubic for a root. z^3 + c2*z^2 + c1*z + c0 = 0
     c2 = 3.0*cmath.sqrt(C)-0.5*A
     c1 = 2.0*C-A*cmath.sqrt(C)
     c0 = -B**2/8.0
     Delta0 = c2**2-3.0*c1
     Delta1 = 2.0*c2**3-9.0*c2*c1+27.0*c0
     Delta = cmath.sqrt(-27.0*(18.0*c2*c1*c0-4.0*c2**3*c0+c2**2*c1**2-4.0*c1**3-27.0*c0**2))
     C2 = (0.5*(Delta1+Delta))**(1.0/3.0)
     z = -1.0/3.0*(c2+C2+Delta0/C2)
     #3. Complete the square and solve for first two roots
     D = cmath.sqrt(2.0*cmath.sqrt(C)-A+2.0*z)
     E = cmath.sqrt(2.0*cmath.sqrt(C)*z+z*z)
     b1 = a3/(2.0*a4)+D
     c1 = a3**2/(16.0*a4**2)+D*a3/(4.0*a4)+E+z+cmath.sqrt(C)
     #3a. factor quartic polynomial into two quadratics
     factor2 = polydiv((a0,a1,a2,a3,a4),(c1,b1,1))[0]
     b2 = factor2[1]/factor2[2]
     c2 = factor2[0]/factor2[2]
     #4. Examine roots for each quadratic, find real positive ones.
     tau = C_limit
     d = (1-C_limit*ab)**2-aa*bb*C_limit**2
     if abs(d)>=epsilon:
        q = a4*C_limit**4+a3*C_limit**3+a2*C_limit**2+a1*C_limit+a0
        obj = ((A42*C_limit**2+A32*C_limit+A22)*C_limit**2+C_limit*max(0,q))/(d**2)
     else: # we have a problem, d=0 at C_limit
        obj = float("inf")
     optimal = (C_limit,obj)
     roots = [0.5*(-b1+cmath.sqrt(b1*b1-4.0*c1)),0.5*(-b1-cmath.sqrt(b1*b1-4.0*c1)),0.5*(-b2+cmath.sqrt(b2*b2-4.0*c2)),0.5*(-b2-cmath.sqrt(b2*b2-4.0*c2))]
     #roots = [r.real for r in roots if abs(r.imag)<epsilon and r.real>0 and r.real<C_limit]
     roots = [r.real for r in roots if abs(r.imag)<epsilon and r.real>0]
     if Verbose and len(roots)>0:
       min_root += min(roots)
       max_root += max(roots)
       max_root_denom += 1.0
     roots = [r for r in roots if r<C_limit]
     for r in roots:
       d = (1-r*ab)**2-aa*bb*r**2
       obj = (A42*r**2+A32*r+A22)*r**2/(d**2)
       if obj<optimal[1]:
         optimal = (r,obj)
     if optimal[1]<float("inf"):
       tau = optimal[0]
       d = (1-tau*ab)**2-aa*bb*tau**2
       updateL = tau/d*((1.0-tau*ab)*(La*b+Lb*a)+(tau*bb*La)*a+(tau*aa*Lb)*b)
       L = L + updateL
   if Verbose:
     print "Iteration "+str(k+1)+" complete."
     if max_root_denom>0.0: print "Average loss: "+str(totloss/float(batch_size))+",  avg roots: ["+str(min_root/max_root_denom)+", "+str(max_root/max_root_denom)+"]"
   if totloss < batch_size*loss_tol:
     break
 loss = totloss/float(batch_size)
 loss_sig = np.sqrt((totloss2-2*loss*totloss+batch_size*loss**2)/float(batch_size-1))
 lossCI = 1.96*loss_sig/np.sqrt(batch_size)
 if totloss < batch_size*loss_tol:
   print "Stopping criterion met in "+str(k+1)+" iterations. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
 else:
   print "Maximum number of iterations ("+str(itmax)+") exceeded. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
 return(L)



##################### Tests #####################

def mahalanobis(x,y,**kwargs):
    xd=np.matrix(x-y)
    try:
        return xd*kwargs["A"]*xd.T
    except:
        #we have to cheat since scikit seems to check the function but doesn't use the correct dimensions
        print 'except'
        return np.linalg.norm(x-y)

def simKNNpredict(X,Y,Xtest,Ytest,L,k=3,weighted=True,method='brute'):
    if not isinstance(X, np.matrix):
        X=np.matrix(X)
    if not isinstance(Xtest, np.matrix):
        Xtest=np.matrix(Xtest)
    #transform data
    X=X*L.T
    Xtest=(L*Xtest.T)
    n=Xtest.shape[1]
    y_pred = np.empty(n, dtype=Ytest.dtype)
    if method=='tree':
        #build tree
        print 'not implemented'
    else:
        #resort to brute force
        n=Xtest.shape[1]
        for x in xrange(n):
            w=np.squeeze(np.asarray(X*Xtest[:,x]))
            neighb=np.argpartition(-w, k)[:k]
            #cannot use bottleneck unless numpy version >= 1.9 available
            #argpartsort(np.squeeze(np.asarray(X*Xtest[:,x])),k)[:k]

            #unweighted
            #lmode, num=mode(Y[neighb], axis=0)
            #weighted
            lmode, weight = weighted_mode(Y[neighb], w[neighb])
            y_pred[x]=lmode
    return y_pred


def test():
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import StratifiedKFold
    digits = load_digits()
    X=digits.data
    Y=digits.target
    # 10-fold cross validation
    '''
    skf = StratifiedKFold(Y, 10,random_state=17)
    res=[]
    print 'Symmetric OASIS kernel'
    for j,x in enumerate(skf):
        #Kyle kernel
        L=OASIS_SIM(10,X[x[0]],Y[x[0]],C=1e-5,itmax=200,batch_size=100,loss_tol=1e-3,epsilon=1e-12)
        #We cannot use sklearn knn since it requires a metric (distance, not similairty). We therefore use our own
        pred=simKNNpredict(X[x[0]],Y[x[0]],X[x[1]],Y[x[1]],L,k=3,weighted=False,method='brute')
        acc=float(sum(pred==Y[x[1]]))/len(Y[x[1]])
        res.append(acc)
        print 'accuracy: '+str(acc)
    res=np.array(res)
    print 'average: '+str(res.mean())
    print 'std dev: '+str(res.std())
    exit()
    skf = StratifiedKFold(Y, 10,random_state=17)
    res=[]
    print 'RDML'
    for j,x in enumerate(skf):
        #RDML
        A=RDML(X[x[0]],Y[x[0]],lmbda=0.2,T=50*len(x[0]))
        clf = KNeighborsClassifier(n_neighbors=1,metric='pyfunc',method='auto',metric_params={"A": A},func=mahalanobis)

        #train
        clf.fit(X[x[0]],Y[x[0]])
        #predict
        acc=clf.score(X[x[1]],Y[x[1]])
        res.append(acc)
        print 'accuracy: '+str(acc)
    res=np.array(res)
    print 'average: '+str(res.mean())
    print 'std dev: '+str(res.std())
    '''
    skf = StratifiedKFold(Y, 10,random_state=17)
    res=[]
    print 'Liu et al.'
    for j,x in enumerate(skf):
        #Liu et al
        L=LowRankBiLinear(10,X[x[0]],Y[x[0]],0.1,10**-6,0.1,0.1,100,tol=1e-6,epsilon=10**-8)
        pred=simKNNpredict(X[x[0]],Y[x[0]],X[x[1]],Y[x[1]],L.T,k=3,weighted=False,method='brute')
        acc=float(sum(pred==Y[x[1]]))/len(Y[x[1]])
        res.append(acc)
        print 'accuracy: '+str(acc)
    res=np.array(res)
    print 'average: '+str(res.mean())
    print 'std dev: '+str(res.std())
    

if __name__ == "__main__":  
    test()
