# Sample solution to assignment 6, CSCI/MATH2072U, Ontario Tech U, 2024, by L. van Veen.
import numpy as np
from NR import NR

# Gradient of the least-squares objective function.
def f(xs,ys,c,lamb):
    S = np.zeros(2,)
    n = np.shape(xs)[0] - 1
    for i in range(n+1):
        S[0] += 2.0 * (c * np.exp(lamb * xs[i]) - ys[i]) * np.exp(lamb * xs[i])
        S[1] += 2.0 * (c * np.exp(lamb * xs[i]) - ys[i]) * c * xs[i] * np.exp(lamb * xs[i])
    return S

# Jacobian matrix of second derivatives of the least-squares objective function.
def Df(xs,ys,c,lamb):
    S = np.zeros((2,2))
    n = np.shape(xs)[0] - 1
    for i in range(n+1):
        S[0,0] += 2.0 * np.exp(2.0 * lamb * xs[i])
        S[0,1] += 2.0 * (2.0 * c * np.exp(lamb * xs[i]) - ys[i]) * xs[i] * np.exp(lamb * xs[i])
        S[1,1] += 2.0 * (2.0 * c * np.exp(lamb * xs[i]) - ys[i]) * c * xs[i]**2 * np.exp(lamb * xs[i])
    S[1,0] = S[0,1]
    return S

# One way to estimate the parameters of the exponantial function, based on the first and last data point.
def setInitialValues(xs,ys):
    n = np.shape(xs)[0] - 1
    lamb = (np.log(ys[n]) - np.log(ys[0])) / (xs[n] - xs[0])
    c = ys[0] * np.exp(-lamb * xs[0])
    print('Initial values are c=%f and lambda=%f.' % (c,lamb))
    return np.array([[c],[lamb]])

# Computation of the least-squares solution by means of Newton-Raphson iteration.
def ELS(xs,ys):
    # Auxiliary definitions of function of a single variable:
    def F(x):
        c = x[0]
        lamb = x[1]
        return f(xs,ys,c,lamb)
    def DF(x):
        c = x[0]
        lamb = x[1]
        return Df(xs,ys,c,lamb)
    # Set the initial guess:
    z0 = setInitialValues(xs,ys)
    # Set the tolerances and max nr of iterations (it may be better to make these inputs for the function!):
    epsz = 1e-8
    epsf = 1e-3
    maxIt = 15
    # Cal Newton-Raphson.
    z,err,res,conv = NR(F,DF,z0,epsz,epsf,maxIt)
    
    return z[0],z[1],conv
