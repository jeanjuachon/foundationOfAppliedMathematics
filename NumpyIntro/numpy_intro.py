#!/usr/bin/env python
# coding: utf-8

# In[101]:


# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name> Jean Philip Juachon
<Class> BUDS
<Date> July 26, 2019
"""
import numpy as np
def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB using the
    dot product function."""
    a = np.array([(3,-1,4),(1,5,-9)])
    b = np.array([(2,6,-5,3),(5,-8,9,7),(9,-3,-2,-3)])
    c = np.dot(a,b)
    return (c)
   


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    a1 = np.array([(3,1,4),(1,5,9),(-5,3,1)])
    answer = -np.dot(a1,np.dot(a1,a1)) + (9*np.dot(a1,a1)) - (15*a1)
    return (answer)
    


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    triangle1 = np.tri(7,7, dtype = int)
    triangleA = np.flip(triangle1)
    fullfive = np.full((7,7),5)
    fullNegOne = np.full((7,7),-1)
    triangleB = np.tril(fullNegOne) + np.triu(fullfive, np.diag([1]))
    product1A = np.dot(triangleA,triangleB)
    product_final = np.dot(product1A, triangleA)
    product_final = product_final.astype(np.int64)
    return product_final
    


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.
    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    array_copy = np.copy(A)
    array_copy[array_copy < 0] = 0
    return array_copy
   


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    Used the reshape, transpose, and stack function, comments are indicated below
    """
    a = np.arange(6)
    reshape = np.reshape(a,(3,2))
    MatrixA = np.transpose(reshape) #Matrix A final variable

    tri1 = np.tri(3,3, dtype = int)
    MatrixB = tri1 * 3 #Matrix B final variable
    
    MatrixC = np.diag([-2,-2,-2]) #Matrix C final Variable
    
    #zero to stack
    zero1 = np.zeros((3,3), dtype = np.int64)
    zero2 = np.zeros((2,2), dtype = np.int64)
    zero3 = np.zeros((3,2), dtype = np.int64)
    zero4 = np.zeros((2,3), dtype = np.int64)
    
    #stacked values
    top = np.hstack((zero1, np.transpose(MatrixA), np.identity(3)))
    mid = np.hstack((MatrixA, zero2, zero4))
    bottom = np.hstack((MatrixB, zero3, MatrixC))
    vstack = np.vstack((top,mid,bottom))
    
    return vstack
   


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.
    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    A = np.array(A) #generate the list
    #m2 = A.reshape(2,int(len(A)/2)) #reshape the list to 2xN
    m3 = A.sum(axis = 1) #sum the rows
    m4 = m3.reshape(-1,1)
    m5 = A / m4
    
    return m5
   


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    Horizontal, Vertical, Diagonal left, Diagonal right
    """
    grid = np.load("grid.npy")
    answer = 0
    z = 0
    for y in range(0,20):
        for x in range(0,17):
            z = grid[y][x] * grid[y][x + 1] * grid[y][x + 2] * grid[y][x + 3]
            if z > answer:
                answer = z
    for y in range(0,20):
        for x in range(0,17):
            z = grid[x][y] * grid[x + 1][y] * grid[x + 2][y] * grid[x + 3][y]
            if z > answer:
                answer = z
    for y in range(0,17):
        for x in range(0,17):
            z = grid[y][x] * grid[y + 1][x + 1] * grid[y + 2][x + 2] * grid[y + 3][x + 3]
            if z > answer:
                answer = z
    for y in range(0,17):
        for x in range(3,20):
            z = grid[y][x] * grid[y + 1][x - 1] * grid[y + 2][x - 2] * grid[y + 3][x - 3]
            if z > answer:
                answer = z

    print("Answer is: ",answer)
    return answer
   




