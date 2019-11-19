# newtons_method.py
"""
Volume 1: Newton's Method.
<Jean Philip L. Juachon>
<Multivariable Calculus - Lab 9: Newton's Method>
September 2019
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


#Prob 1,3,5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """
    Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    if np.isscalar(x0):
        for k in range(maxiter):
            x1 = x0 - alpha*f(x0)/Df(x0)
            if abs(x1 - x0) < tol:
                return x1, True, k+1
            x0 = x1
        
        return x1, False, k+1
    
    else:
        for k in range(maxiter):
            y = la.solve(Df(x0), f(x0))
            x1 = x0 - alpha*y
            if la.norm(x1 - x0) < tol:
                return x1, True, k+1
            x0 = x1
        
        return x1, False, k+1
            
            

#Prob 2
def prob2(N1, N2, P1, P2):
    """
    Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    f = lambda r: P1*((1+r)**N1 - 1) - P2*(1 - (1+r)**(-N2))
    Df = lambda r: P1*N1*(1+r)**(N1-1) - N2*P2*(1+r)**(-N2-1)
    
    return newton(f, 0.1, Df)[0]
    

#Prob 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """
    Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    alpha_arr = np.linspace(0, 1, 1000)
    iters = []
    for a in alpha_arr[1:]:
        iters.append(newton(f, x0, Df, alpha=a)[2])
    iters = np.array(iters)
    
    plt.figure()
    plt.plot(alpha_arr[1:], iters)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Iterations')
    plt.show()
    
    return alpha_arr[np.argmin(iters)]
    

# Problem 6
def prob6():
    """
    Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    f = lambda x: np.array([5*x[0]*x[1] - x[0]*(1+x[1]),
                            -x[0]*x[1] + (1-x[1])*(1+x[1])])
    Df = lambda x: np.array([[4*x[1]-1, 4*x[0]],
                             [-x[1], -x[0] - 2*x[1]]])
    
    true_root1 = np.array([0,1])
    true_root2 = np.array([0,-1])
    true_root3 = np.array([3.75, 0.25])
    tol = 1e-5
    
    while True:
        i = np.random.random()* -0.25
        j = np.random.random()*0.25
        
        root1, conv1 = newton(f, np.array([i,j]), Df, maxiter=50)[0:2]
        root2, conv2 = newton(f, np.array([i,j]), Df, alpha=0.55, maxiter=50)[0:2]

        error1 = la.norm(root1-true_root1)
        error2 = la.norm(root1-true_root2)
        error3 = la.norm(root2-true_root3)
        
        if ((error1 < tol) or (error2 < tol)) and (error3 < tol) and (conv1 == True) and (conv2 == True):
            return np.array([i,j])
            
            
# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """
    Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
        
    Returns:
        
    """
    x_real = np.linspace(domain[0], domain[1], res)
    x_imag = np.linspace(domain[2], domain[3], res)
    
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag
    
    for k in range(iters):
        X_1 = X_0 - f(X_0)/Df(X_0)
        X_0 = X_1
        
    Y = np.zeros((res,res), float)
    for i in range(res):
        for j in range(res):
            Y[i,j] = np.argmin(np.abs(zeros-X_1[i,j]))
            
    plt.pcolormesh(X_real, X_imag, Y, cmap='brg')
    
            
    
    
    
    
    
    
    
