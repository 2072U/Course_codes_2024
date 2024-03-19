# Polynomial least-squares fit to data (xs[i],ys[i]) i=0..m-1 (n by 1 arrays of floats).
# Last input is the polynomial order (positive integer).
# Output: function handle for the polynomial and its coefficients a (o+1 by 1 array of floats).
# By L. van Veen, Ontario Tech U, 2024.
import numpy as np

# In: list of x and y values and order of approximation. Out: function handle for interpolant and list of its coefficients.
def LS(xs,ys,o):
    m = np.size(xs)
    V = np.ones((m,o+1))                            # Construct the coefficient matrix in O(m o) FLOPs.
    for i in range(m):
        for j in range(1,o+1):
            V[i,j] = V[i,j-1] * xs[i]
    a,R,rank,s = np.linalg.lstsq(V,ys,rcond=None)   # Compute the least-squares solution - note that it has multiple outputs.
    print('Relative least-squares residual is %e.' % (R/np.linalg.norm(ys,2))) # Show the relative residual.
    
    def P(z):                                       # Construct the polynomial to return
        S = a[o-1] + a[o]*z                         # Use Horner's algorithm so that P(x) is computed in O(o) time
        for k in range(o-2,-1,-1):
            S = a[k] + S * z
        return S
    return P,a

