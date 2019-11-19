# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<JEAN PHILIP JUACHON>
<BUDS PROGRAM 2019>
<SEPT 2019>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    q,r = la.qr(A, mode = 'economic')
    return la.solve_triangular(r, q.T@b)


# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing = np.array(np.load("housing.npy"))
    ones = np.ones((len(list(housing[:,0])), 2))         
    ones[:,0] = housing[:,0]                              

    b = housing[:,1]                                  

    weights = least_squares(ones, b)                   
    x = np.linspace(min(housing[:,0]), max(housing[:,0]), num=50) 
    y = weights[1] + weights[0]*x                 

    plt.scatter(ones[:,0], b, label="Data Points") 
    plt.plot(x, y, label="Least Squares Fit")       
    plt.legend()
    plt.title("Yearly housing prices")
    plt.ylabel("Prices")
    plt.xlabel("Year(2000's)")
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    housing = np.load("housing.npy")
    #plt.scatter(housing[:,0],housing[:,1])
    A = housing[:,0]
    b = housing[:,1]
    n = 3

    poly_A = np.zeros((n+1,n+1)).astype(float)
    poly_sum = [(np.sum(np.power(A,i))) for i in range(n+n+2)]
    
    poly_b = np.zeros(n+1).astype(float)
    for i in range(0,n+1,1):
        poly_A[i] = poly_sum[i:i+n+1]
    for i in range(0,n+1):
        val = 0
        for j in range(0,len(b)):
            val += (b[j]*A[j]**i)
        poly_b[i] = (val)      
    
    lsq= la.lstsq(poly_A, poly_b)[0]
    
    values = []
    for i in A:
        val = lsq[0]
        for j in range(1,len(lsq)):
            val += lsq[j]*i**j
        values.append(val)
    plt.scatter(A,b)
    plt.scatter(A,values)
    plt.show()


    n = 6

    poly_A = np.zeros((n+1,n+1)).astype(float)
    poly_sum = [(np.sum(np.power(A,i))) for i in range(n+n+2)]
    
    poly_b = np.zeros(n+1).astype(float)
    for i in range(0,n+1,1):
        poly_A[i] = poly_sum[i:i+n+1]
    for i in range(0,n+1):
        val = 0
        for j in range(0,len(b)):
            val += (b[j]*A[j]**i)
        poly_b[i] = (val)      
    
    lsq= la.lstsq(poly_A, poly_b)[0]
    
    values = []
    for i in A:
        val = lsq[0]
        for j in range(1,len(lsq)):
            val += lsq[j]*i**j
        values.append(val)
    plt.scatter(A,b)
    plt.scatter(A,values)
    plt.show()

    n = 9

    poly_A = np.zeros((n+1,n+1)).astype(float)
    poly_sum = [(np.sum(np.power(A,i))) for i in range(n+n+2)]
    
    poly_b = np.zeros(n+1).astype(float)
    for i in range(0,n+1,1):
        poly_A[i] = poly_sum[i:i+n+1]
    for i in range(0,n+1):
        val = 0
        for j in range(0,len(b)):
            val += (b[j]*A[j]**i)
        poly_b[i] = (val)      
    
    lsq= la.lstsq(poly_A, poly_b)[0]
    
    values = []
    for i in A:
        val = lsq[0]
        for j in range(1,len(lsq)):
            val += lsq[j]*i**j
        values.append(val)
    plt.scatter(A,b)
    plt.scatter(A,values)
    plt.show()



def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    data = np.array(np.load("ellipse.npy")) 
    x_data = data[:,0]                     
    y_data = data[:,1]                    
    b = np.ones((len(list(data[:,1])), 1))  
    A = np.ones((len(list(data[:,0])), 5))

  
    A[:,0] = x_data**2
    A[:,1] = x_data
    A[:,2] = x_data*y_data
    A[:,3] = y_data
    A[:,4] = y_data**2

    coeffs = la.lstsq(A, b)[0]                                          
    plt.scatter(data[:,0], data[:,1])                                   
    plot_ellipse(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]) 
    plt.show()     



# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = np.shape(A)
    x = np.random.random(n)
    x = x/la.norm(x)
    e = x.T  @ np.dot(A,x)
    for k in range(1,N):
        x_new = np.dot(A,x)
        x_new = x_new/la.norm(x_new)
        x = x_new
        e1 = x.T  @ np.dot(A,x)
        if np.abs(e1-e) < tol:
            break
        e = e1
    
    return (x.T @ np.dot(A,x),x)



# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = np.shape(A)
    S = la.hessenberg(A)
    for k in range(0,N):
        Q,R = la.qr(S)
        S = (R @ Q)
    eigs = []
    i = 0
    while i < n: # Assume s1 = 1 x 1
        eigs.append(S[i,i])
        i = i + 1
    return(eigs)

