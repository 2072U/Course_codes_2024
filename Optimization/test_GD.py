# Test script for gradient descent, using the problem for A6.
# CSCI/MATH2072U, 2024, lecture 23. Ontario Tech U, by L. van Veen.
import numpy as np
from functions import *
from GD import GD
import matplotlib.pyplot as plt

# Generate test data. Exponential dependance with some noise with amplitude eps.
c = 1.5
lamb = 0.4
L = 0.0
R = 10.0
n = 100
eps = 0.01
xs, ys = generateData(n,c,lamb,L,R,eps) # n points between x=L and x=R.

# Auxiliary functions of a single, vextor-valued variable.
def F(z):
    c = z[0]
    lamb = z[1]
    return f(c,lamb,xs,ys,n)
def gradF(z):
    c = z[0]
    lamb = z[1]
    return gradf(c,lamb,xs,ys,n)

# Initial guess. Gradient descent is much less sensitive to the initial guess.
z0 = np.array([1.0,1.0])

maxIt = 100                       # Max nr. of iterations.
eps = 1e-3                        # Convergence criterion. 
z, errs = GD(z0,F,gradF,maxIt,eps)# Call gradient descent.
# Plot the results: cost function, gain and step size.
c_gd = z[0]
lamb_gd = z[1]
print('Original c=%f; result of gradient descent is %f.' % (c,c_gd))
print('Original lambda=%f; result of gradient descent is %f.' % (lamb,lamb_gd))
plt.semilogy(errs[:,0],errs[:,1],'-*')
plt.xlabel('nr of iterations')
plt.ylabel('cost function')
plt.show()
plt.plot(errs[:,0],errs[:,2],'-*')
plt.xlabel('nr of iterations')
plt.ylabel('gain (old res - new res)/old_res')
plt.show()
plt.plot(errs[:,0],errs[:,3],'-*')
plt.xlabel('nr of iterations')
plt.ylabel('step size')
plt.show()
