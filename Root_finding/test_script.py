# Test script for bisection and Newton iteration, function taken from Assignment 2.
# By L. van Veen, Ontario Tech U, 2023.
import numpy as np
from bisect import *
from Newton import *
import matplotlib.pyplot as plt

def f(x):
    return (x-1.0) * np.cos(x)+0.5
def dfdx(x):
    return np.cos(x) - (x-1.0) * np.sin(x)

# First bisection:
a = 0.0
b = 1.0
kMax = 20
epsX = 1e-13
epsF = 1e-13

x, log_b = bisect(f,a,b,kMax,epsX,epsF)

# Then Newon iteration:
x0 = 17.0
kMax =12
epsX = 1e-13
epsF = 1e-13

x, log_N = Newton(f,dfdx,x0,kMax,epsX,epsF)

plt.semilogy(log_b[:,0],log_b[:,2],'r-*')
plt.semilogy(log_N[:,0],log_N[:,2],'b-*')
plt.xlabel('nr. of iterations')
plt.ylabel('residual')
plt.show()

