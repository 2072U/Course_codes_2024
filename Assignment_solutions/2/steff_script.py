# Author: L. van Veen, UOIT 2023. Solutions for assignment 2, Q2.
import numpy as np
from steff import steff
from newton import newton

# Question 2
def f(x):
    return np.exp(x-x**2)-x/2.0-1.0836
def fp(x):
    return (1.0-2.0*x)*np.exp(x-x**2)-0.5

# residual and error tolerance, number of iterations
tol_err = 1e-10
tol_res = 1e-10
itmax = 10

# initial point
x0 = 1.0

# First try Newon iteration
print("Calling Newton function...")
x,err,res = newton(f,fp,x0,tol_res,tol_err,itmax)
# That converges slowly, now try steffenson
print("Calling Steffensen function...")
x,err,res = steff(f,fp,x0,tol_err,tol_res,itmax)
