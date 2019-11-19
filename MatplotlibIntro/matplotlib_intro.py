#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>Jean Philip L. Juachon
<Class>
<Date>Aug 3, 2019
"""
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.
    Parameters:
        n (int): The number of rows and columns in the matrix A.
    Returns:
        (float) The variance of the means of each row.
    """
    a = np.random.normal(size = (n,n))
    row_mean1 = np.mean(a, axis = 1)
    return np.var(row_mean1)

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    variances = []
    for element in range(1,11):
        variances.append(var_of_means(element * 100))
    variances = np.array(variances)
    plt.plot(100*np.arange(1,11),variances)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    a = np.pi #define pi
    x = np.linspace(-2*a,2*a) #define -2pi and 2pi
    plt.plot((x), np.sin(x)) #plot
    plt.plot((x), np.cos(x))
    plt.plot((x), np.arctan(x))
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    domain = np.linspace(-2,6)
    y = 1/(domain - 1)
    plt.plot(domain[:19], y[:19], "m--", linewidth = 4)
    plt.plot(domain[19:], y[19:], "m--", linewidth = 4)
    plt.title("Plot of f(x) = 1/(x-1)")
    plt.xlim(-2,6)
    plt.ylim(-6,6)


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    pi = np.pi
    x = np.linspace(0, 2*pi)

    y = np.sin(x)
    x1 = plt.subplot(221)
    plt.axis([0,2*pi,-2,2])
    plt.xlabel("sin pi")
    x1.plot(x,y,"g-")

    y2 = np.sin(2*x)
    plt.xlabel("sin 2pi")
    x2 = plt.subplot(222)
    plt.axis([0,2*pi,-2,2])
    x2.plot(x,y2,"r--")

    y3 = 2*(np.sin)(x)
    x3 = plt.subplot(223)
    plt.axis([0,2*pi,-2,2])
    plt.xlabel("2sin pi")
    x3.plot(x,y3,"b--")

    y4 = 2*(np.sin)(2*x)
    x4 = plt.subplot(224)
    plt.xlabel("2sin2pi")
    plt.axis([0,2*pi,-2,2])
    x4.plot(x,y4,"m:")
    plt.suptitle("Different plots of sin")
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    #getting the fars data
    fars = np.load("FARS.npy")
    hrs = fars[:,0]
    long = fars[:,1]
    lat = fars[:,2]
    
    #scatter plot
    plot2 = plt.subplot(121)
    plot2.plot(long,lat,  "k,", color = "black")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plot2.set_aspect("equal")
    
    #histogram
    plot1 = plt.subplot(122)
    plot1.hist(hrs, bins = np.arange(0,25))
    plt.xlim(0,24)
    plt.xlabel("Time")
    plt.show()


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    pi = np.pi
    x, y = [-2*pi, 2*pi], [-2*pi,2*pi] #domain of -2pi to 2pi for x and y.
    X, Y = np.meshgrid(x, y) # Combine the 1-D data into 2-D data.
    
    x = np.linspace(-2*pi, 2*pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y) / (X*Y)
    
    # Plot the heat map of f over the 2-D domain.
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="magma")
    plt.colorbar()
    plt.xlim(-2*pi, 2*pi)
    plt.ylim(-2*pi, 2*pi)
    
    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap="coolwarm")
    plt.colorbar()
    #raise NotImplementedError("Problem 6 Incomplete")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




