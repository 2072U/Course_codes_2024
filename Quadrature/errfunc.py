# The error function, i.e. the antiderivative (with left limit 0 and right limit 1) of the bell curve (Gaussian distribution) approximated by the composite Simpson's rule.
# CSCI/MATH2072U, Ontario Tech U, 2024, by L. van Veen.
import numpy as np

# Define the normal (Gaussian) distribution.
def normal(x,mu,sig):
    return np.exp(-(x-mu)**2/(2.0*sig**2))/np.sqrt(2.0*np.pi*sig**2)

# Composite Simpson's rule with M sub intervals in the domain [L,R] for function f.
def cS(L,R,f,M):
    h = (R-L) / float(M)                        # Compute the sub interval width.
    I = 0.0                                     # Initialize the integral I.
    for k in range(M):                          # Loop over sub intervals.
        a = L + float(k) * h                    # Compute the left boundary of the current sub interval ...
        b = L + float(k+1) * h                  # and the right boundary ...
        m = (a + b) / 2.0                       # and the mid point.
        I += (f(a) + 4.0 * f(m) + f(b)) * h/6.0 # Simpson's rule, based on a local quadratic approximation of f.
    return I

# Now use this quadrature to approximate the error function int(normal(x,mu,sig), x=l..x).
def errf(x,l,mu,sig,M):
    # Auxiliary definition of a function of one variable (the bell curve/normal or Gaussian distribution).
    def f(x):
        return normal(x,mu,sig)
    # Compute the integral with the composite Simpson's rule.
    R = cS(l,x,f,M)
    return R
