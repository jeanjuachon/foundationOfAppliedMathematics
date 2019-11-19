#!/usr/bin/env python
# coding: utf-8
"""
<JEAN PHILIP JUACHON>
<BUDS PROGRAM 2019>
<SEPT 2019>
"""
# In[1]:


import numpy as np
from scipy import linalg as la


# In[122]:


#Prob1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n,n))
    for i in range(0,n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1, n-1):
            R[i,j] = (Q[:,j].T) @ Q[:,i]
            Q[:,j] -= R[i,j]*Q[:,i]
    return Q,R
    #raise NotImplementedError("Problem 1 Incomplete")


# In[117]:


#A = np.random.random((6,4))
#print(qr_gram_schmidt(A))
#print(la.qr(A))


# In[118]:


#Prob 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    q,r = la.qr(A)
    return np.abs(np.prod(np.diag(r)))
    #raise NotImplementedError("Problem 2 Incomplete")


# In[119]:


A = np.array(([1,2,3],[1,2,4],[2,4,6]))
print(abs_det(A))

print(la.det(A))


# In[120]:


def back_substitution(A,b):
    n = b.size
    x = np.zeros_like(b).astype(float)

    for i in range(n-1, -1, -1):
        val = np.sum(A[i]) - A[i,i]
        x[i] = np.divide(b[i]-val,A[i,i])
        for j in range (i-1, -1, -1):
            A[j,i] = A[j,i]*x[i]
    return x 
      
#prob3    
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    q,r = la.qr(A)
    y = np.dot((q.T),b)
    x = back_substitution(r, y)
    return x
    #raise NotImplementedError("Problem 3 Incomplete")


# In[121]:


#A = np.array(([1,2,3],[1,4,5],[0,0,6]))
#b = np.array([2,2,3])
#la.solve(A, b)
#print(solve(A,b))


# In[102]:


#prob4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m[,]n) ndarray): A matrix of rank n.

    Returns:r
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x >= 0 else -1
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.eye(m)                                              

    
    for k in range(n):
        u = np.copy(R[k:,k])
        u[0] = u[0] + (sign(u[0]) * la.norm(u))
        u = u / (1.0 * la.norm(u))                               
        R[k:,k:] = R[k:,k:] - (2*np.outer(u, (u.T @ R[k:,k:])))
        Q[k:,:] = Q[k:,:] - (2*np.outer(u, (u.T @ Q[k:,:])))     
    
    return Q.T, R
    #raise NotImplementedError("Problem 4 Incomplete")


# In[103]:


#A = np.random.random((5, 3))
#qr_householder(A)


# In[115]:


def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x : 1 if x >= 0 else -1
    m,n = np.shape(A)
    H = np.copy(A).astype(np.float64)
    Q = np.eye(m).astype(np.float64)
    for k in range(n-2):
        u = np.copy(H[k+1:,k]).astype(np.float64)
        u[0] += sign(u[0]) * np.linalg.norm(u)
        u /= np.linalg.norm(u)
        H[k+1:,k:] -= 2*np.outer(u, (u.T @ H[k+1:,k:]))
        H[:,k+1:] -= 2*np.outer(H[:,k+1:] @ u, u.T)
        Q[k+1:,:] -= 2*np.outer(u, (u.T @ Q[k+1:,:]))     
    return H, Q.T

