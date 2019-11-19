#!/usr/bin/env python
# coding: utf-8

# In[237]:


from scipy import linalg as la
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import time
from scipy import sparse
from scipy.sparse import linalg as spla


# In[257]:


# gmres.py
"""Volume 1: GMRES.
<Name: Jean Philip Juachon>
<Class: BUDS 2019>
<Date: 06/11/2019>
"""
# Problems 1 and 2.
def gmres(A, b, x0, k=100, tol=1e-8, plot=False):
    """Calculate approximate solution of Ax=b using the GMRES algorithm.

    Parameters:
        A ((m,m) ndarray): A square matrix.
        b ((m,) ndarray): A 1-D array of length m.
        x0 ((m,) ndarray): The initial guess for the solution to Ax=b.
        k (int): Maximum number of iterations of the GMRES algorithm.
        tol (float): Stopping criterion for size of residual.
        plot (bool): Whether or not to plot convergence (Problem 2).

    Returns:
        ((m,) ndarray): Approximate solution to Ax=b.
        res (float): Residual of the solution.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    #INITIALIZATION
    Q = np.empty((np.size(b), k+1))
    H = np.zeros((k+1, k))
    r0 = b - A @ x0
    Q[:,0] = r0 / la.norm(r0)
    beta = la.norm(r0)
    resid = []
    eigen = la.eig(A)[0]
    realEig = eigen.real
    fakeEig = eigen.imag
    iterations = []
    #be1[0] = beta
    
    for j in range(k):
        iterations.append(j+1)
        Q[:,j+1] = A @ Q[:,j]
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i].T , Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        H[j+1,j] = la.norm(Q[:,j+1])
        if abs(H[j+1,j]) > tol:    #absolute value to be sure that it's not zero
            Q[:,j+1] /= H[j+1,j]
        be1 = np.zeros(j + 2)
        be1[0] = 1
        #leastSquares    
        #y, r = la.lstsq(H[:j+2,:j+1], be1[:j+2])[:2]
        #y = la.lstsq(H[:j+2,:j+1], be1[:j+2])[0]
        y = la.lstsq(H[:j+2,:j+1], beta*be1)[0]

        #resid
        #resid = np.sqrt(y[0])
        resid.append(la.norm(H[:j+2,:j+1] @ y - beta*be1))
        if resid[j] < tol:
            if plot:
                plot1 = plt.subplot(121)
                plot1.scatter(realEig, fakeEig)
                plot2 = plt.subplot(122)
                plot2.semilogy(iterations, resid)
                plt.show()
            return Q[:,:j+1] @ y + x0, resid[j]
    if plot: #if it didn't converege, the plot is:
        plot1 = plt.subplot(121)
        plot1.scatter(realEig, fakeEig)
        plot2 = plt.subplot(122)
        plot2.semilogy(iterations, resid)
        plt.show()    
    return Q[:,:j+1] @ y + x0, resid[k-1]





# In[258]:


A = np.array([[1,0,0],[0,2,0],[0,0,3]])
b = np.array([1, 4, 6])
x0 = np.zeros(b.size)
gmres(A, b, x0, k=100, tol=1e-8, plot = True)


# In[225]:


"""k = 100
tol = 1e-8
Q = np.empty((np.size(b), k+1))
H = np.zeros((k+1, k))
r0 = b - A @ x0
Q[:,0] = r0 / la.norm(r0,2)
beta = la.norm(r0)
be1 = np.zeros(k + 1)
be1[0] = beta
for j in range(k-1):
        Q[:,j+1] = A @ Q[:,j]
        for i in range(j):
            H[i,j] = np.dot(Q[:,i],Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        H[j+1,j] = la.norm(Q[:,j+1],2)
        if H[j+1,j] > tol:  
            Q[:,j+1] /= H[j+1,j]
            
        y, r = la.lstsq(H[:j+2,:j+1], be1[:j+2])[:2]
        
        resid = np.sqrt(y[0])"""


# In[130]:





# In[155]:


# Problem 3
def prob3(m=200):
    """For n=-4,-2,0,2,4 create a matrix A= n*I + P where I is the mxm
    identity, and P is an mxm matrix with entries drawn from a normal
    distribution with mean 0 and standard deviation 1/(2*sqrt(m)).
    For each of the given values of n call gmres() with A, a vector of ones called b, an initial guess x0=0, and plot=True

    Parameters:
        m (int): Size of the matrix A.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    for n in (-4, -2, 0, 2, 4):
        b = np.ones(m)
        x0 = 0 * b
        A = n * np.eye(m) + np.random.normal(0, .5/np.sqrt(m), (m,m))
        gmres(A, b, x0, plot=True)
    
    print("It converges slower when the e-vals are clustered around the origin")

# In[156]:


prob3()


# In[260]:


# Problem 4
def gmres_k(A, b, x0, k=5, tol=1E-8, restarts=50):
    """Implement the GMRES algorithm with restarts. Terminate the algorithm
    when the size of the residual is less than tol or when the maximum number
    of restarts has been reached.

    Parameters:
        A ((m,m) ndarray): A square matrix.
        b ((m,) ndarray): A 1-D array of length m.
        x0 ((m,) ndarray): The initial guess for the solution to Ax=b.
        k (int): Maximum number of iterations of the GMRES algorithm.
        tol (float): Stopping criterion for size of residual.
        restarts (int): Maximum number of restarts. Defaults to 50.

    Returns:
        ((m,) ndarray): Approximate solution to Ax=b.
        res (float): Residual of the solution.
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    r = 0
    
    while r <= restarts:
        y, resid = gmres(A, b, x0, k, tol)
        if resid < tol:
            return y, resid
        else:
            x0 = y
            r += 1
            
    return y, resid


# In[214]:


A = np.array([[1,0,0],[0,2,0],[0,0,3],])
b = np.array([1, 4, 6])
x0 = np.zeros(b.size)
gmres_k(A, b, x0, k=5, tol=1e-8, restarts = 50)


# In[280]:


# Problem 5
def time_gmres(m=200):
    """Using the same matrices as in problem 2, plot the time required to
    complete gmres(), gmres_k(), and scipy.sparse.linalg.gmres() with
    restarts. Plot the values of n against the times.

    Parameters:
        m (int): Size of matrix A.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    prob1 = []
    prob2 = []
    fromScipy = []
    sequence = np.arange(25, m+1, 25)
    for m in sequence:   
        b = np.ones(m)
        x0 = 0 * b
        P = np.random.normal(0, .5/np.sqrt(m), (m,m))
        A = m * np.eye(m) + P

        t1 = time.time()
        y1, res = gmres(A,b,x0, k = 100, tol = 1e-8)
        t2 = time.time()
        prob1.append(t2-t1)
        
        y2, res = gmres_k(A, b, x0, k = 5, tol = 1e-8, restarts = 50)
        t3 = time.time()
        prob2.append(t3-t2)
        
        y3, res = spla.gmres(A, b, restart = 1000)
        t4 = time.time()
        fromScipy.append(t4-t3)

        print ("n = ",m)

        print ("GMRES:  ", t2-t1)


        print ("GMRES(k):",t3-t2)

        print ("LA.GMRES = ",t4-t3)
    plt.figure()
    plt.title("Running time comparisons of GMRES, GMRES(K), AND SPLA.GMRES")
    plt.gca().legend(('prob1','prob2','fromScipy'))
    plt.plot(sequence, prob1, label = "Prob 1 time")
    plt.plot(sequence, prob2, label = "GMRES_K TIME")
    plt.plot(sequence, fromScipy, label = "SCIPY TIME")
    plt.ylabel("Time in Secs")
    plt.xlabel("Number of N")
    plt.xticks([25,50,75,100,125,150,175,200])
    plt.show()


# In[281]:


time_gmres()


# In[242]:


#from scipy import sparse
#from scipy.sparse import linalg as spla
#A = np.random.rand(300, 300)
#b = np.random.random(300)
#x, info = spla.gmres(A, b, restart = 1000)
#print(la.norm((A @ x) - b))

#gmres(A,b,x,restart = 1000)
#gmres_k(A,b,x,restarts = 1000)
#print(la.norm((A @ x) - b))


# In[243]:


#help(spla.gmres)


# In[ ]:




