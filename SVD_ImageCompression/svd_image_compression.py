#!/usr/bin/env python
# coding: utf-8
"""
<JEAN PHILIP JUACHON>
<BUDS PROGRAM 2019>
<SEPT 2019>
"""
# In[16]:
from scipy.sparse.linalg import eigsh as la_eigsh
from scipy.linalg import svd as sp_svd
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

# In[17]:
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.
    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.
    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    Ah = A.conj().T                     
    evals, evecs = la_eigsh(Ah @ A, which="SM")      
    
    sigs = np.sqrt(evals)             
    sigs = -1.0 * np.sort(-1.0 * sigs)   
    evecs = -1.0 * np.sort(-1.0 * evecs)
    
    r = np.sum(sigs >= tol)    
    pos_sigs = sigs[:r]        
    pos_sig_evecs = evecs[:,:r] 

    U = (A @ pos_sig_evecs) / pos_sigs 

    return U, pos_sigs, pos_sig_evecs.conj().T


# In[18]:


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    thetas = np.linspace(0, 2*np.pi, 200)
    xs = np.cos(thetas)                  
    ys = np.sin(thetas)                   
    S = np.vstack((xs,ys))               
    E = np.array([[1,0,0], [0,0,1]])     
    U, Sig, Vh = sp_svd(A)                

    Sig = np.diag(Sig) 

   
    plt.subplot(221)
    plt.plot(xs, ys)         
    plt.plot(E[0,:], E[1,:])
    plt.axis("equal")

    VhS = Vh @ S
    VhE = Vh @ E 

  
    plt.subplot(222)
    plt.plot(VhS[0,:], VhS[1,:])
    plt.plot(VhE[0,:], VhE[1,:])
    plt.axis("equal")

    SigVhS = Sig @ VhS
    SigVhE = Sig @ VhE 

    
    plt.subplot(223)
    plt.plot(SigVhS[0,:], SigVhS[1,:])
    plt.plot(SigVhE[0,:], SigVhE[1,:])
    plt.axis("equal")

    USigVhS = U @ SigVhS
    USigVhE = U @ SigVhE

    plt.subplot(224)
    plt.plot(USigVhS[0,:], USigVhS[1,:]) 
    plt.plot(USigVhE[0,:], USigVhE[1,:]) 
    plt.axis("equal")

    plt.show()


# In[19]:


#problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.
    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.
    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, Sig, Vh = sp_svd(A, full_matrices=False)

    if np.linalg.matrix_rank(A) < s:
        raise ValueError("Error: s is greater than the number of nonsingular values of A.")
    
    U_hat = U[:,:s]
    Vh_hat = Vh[:s,:]

    #svd_approximation
    As = U_hat @ np.diag(Sig[:s]) @ Vh_hat
    num_entries = U_hat.size + len(Sig[:s]) + Vh_hat.size

    return As, num_entries


# In[11]:


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, Sig, Vh = sp_svd(A, full_matrices=False)
    if err <= Sig.min():
        raise ValueError("Error: err is less than or equal to the smallest singular value of A.")

    s = np.sum(Sig>err)

    U_hat = U[:,:s]
    Sig_hat = Sig[:s]   
    Vh_hat = Vh[:s,:]
    
    As = U_hat @ np.diag(Sig_hat) @ Vh_hat
    num_entries = U_hat.size + np.shape(Sig_hat)[0] + Vh_hat.size

    return As, num_entries



# In[14]:


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    color = None
    As = None      
    orig_img = imread(filename)/255 
    orig_shape = orig_img.shape     

    if len(list(orig_shape)) == 3:             
        color = True                            
        Rs, sr = svd_approx(orig_img[:,:,0], s) 
        Gs, sg = svd_approx(orig_img[:,:,1], s)
        Bs, sb = svd_approx(orig_img[:,:,2], s)  
        np.clip(Rs, 0, 1)
        np.clip(Gs, 0, 1)
        np.clip(Bs, 0, 1)    
        As = np.dstack((Rs, Gs, Bs))

    else:                                
        color = False                    
        As, so = svd_approx(orig_img, s) 
   
    As = np.reshape(As, orig_shape)

    if color:
        plt.subplot(121)
        plt.imshow(orig_img)
        plt.axis("off")      

        plt.subplot(122)
        plt.imshow(As)       
        plt.axis("off")      

    else: 
        plt.subplot(121)
        plt.imshow(orig_img, cmap="gray") 
        plt.axis("off")                   

        plt.subplot(122)
        plt.imshow(As, cmap="gray")     
        plt.axis("off")                   
    
    plt.show()
    #raise NotImplementedError("Problem 5 Incomplete")


# In[25]:

