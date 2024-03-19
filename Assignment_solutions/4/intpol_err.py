# Example solutions to A4Q2, CSCI/MATH2072U, Ontario Tech U, 2024, by L. van Veen.
import numpy as np
import matplotlib.pyplot as plt
from poly_int import poly_int   # From the course_codes repository.

# Define the test functions and the error bounds:
def f(x):
    print(x)
    return 1.0/(x+1.0)
def g(x):
    return np.exp(-x)
def err_bound_f(N):
    return 0.5**(N+1)
def err_bound_g(N):
    return 0.5**(N+1)/np.math.factorial(N+1)

# The domain of interpolation:
x0 = 0.0
x1 = 0.5
# A fine grid for comparing the function to the interpolant:
m = 1000
x_out = np.linspace(x0,x1,m)
# Range of the test loop:
N_start = 3
N_end = 10
# Pre-allocate an array for the results:
err_f = np.zeros((N_end-N_start+1,3))
err_g = np.zeros((N_end-N_start+1,3))

for N in range(N_start,N_end+1):
    # Generate the interpolation data:
    xs = np.linspace(x0,x1,N+1)
    ys = f(xs)
    y_out = poly_int(xs,ys,x_out)
    err_f[N-N_start,:] = [N,np.max(np.abs(f(x_out)-y_out)),err_bound_f(N)]
    ys = g(xs)
    y_out = poly_int(xs,ys,x_out)
    err_g[N-N_start,:] = [N,np.max(np.abs(g(x_out)-y_out)),err_bound_g(N)]
plt.semilogy(err_f[:,0],err_f[:,1],'-b*',label='approx err for f')
plt.semilogy(err_g[:,0],err_g[:,1],'-g*',label='approx err for g')
plt.semilogy(err_f[:,0],err_f[:,2],'-r',label='max err for f')
plt.semilogy(err_g[:,0],err_g[:,2],'-k',label='max err for g')
plt.legend()
plt.xlabel('Order of interpolation')
plt.ylabel('Interpolation error')
plt.show()

    
