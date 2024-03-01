# Example solution to assignmet A3Q3, MATH/CSCI2072U, Ontario Tech U, 2024, by L. van Veen.
# Test script for multiplication of a banded, circulant matrix with a vector.
import numpy as np
import matplotlib.pyplot as plt
from time import time
from circ_prod import *

eps = 1e-12                    # Tolerance for the norm of the difference between to result of the two methods.
k0 = 2                         # First and last matrix size to try are 2**k0 and 2**k1.
k1 = 14 
wtimes = np.zeros((k1-k0+1,3)) # Allocate array for the wall times.
for k in range(k0,k1+1):       # Loop over sizes.
    n = 2**k                   # Set matrix size.
    wtimes[k-k0,0] = float(n)  # Enter size in array.
    a = np.random.rand(n,1)    # Draw random arrays that are the nonzero entries of the matrix.
    b = np.random.rand(n,1)
    c = np.random.rand(n,1)
    x = np.random.rand(n,1)
    start = time()             # Start the timer.
    y0 = circ_prod(a,b,c,x)    # Call our own function.
    elapsed = time() - start   # Stop the timer and store the wall time.
    wtimes[k-k0,1] = elapsed
    A = np.zeros((n,n))        # Now form the whole array, mostly zeros.
    for i in range(n):
        A[i,i] = a[i]
    for i in range(n-1):
        A[i,i+1] = c[i]
        A[i+1,i] = b[i+1]
    A[0,n-1] = b[0]
    A[n-1,0] = c[n-1]
    start = time()             # Start the timer.
    y1 = A @ x                 # Call the built-in matrix-vector product function.
    elapsed = time() - start   # Stop the timer and store the wall time.
    wtimes[k-k0,2] = elapsed
    # Check that that products match:
    check = np.linalg.norm(y0-y1,2)  # Check if our product agrees with that computed by the built-in function:
    if check > eps:
        print("Products differ! ||y0-y1||=%e for matrix size %d" % (check,n))

plt.loglog(wtimes[:,0],wtimes[:,1],'-*k',label='sparse product')
plt.loglog(wtimes[:,0],wtimes[:,2],'-*b',label='regular product')
plt.loglog(wtimes[:,0],1e-5 * wtimes[:,0],'-g',label='order n')
plt.loglog(wtimes[:,0],1e-9 * wtimes[:,0]**2,'-r',label='order n^2')
plt.xlabel('matrix size n')
plt.ylabel('wall time')
plt.ylim(1e-6,1.0)             # Note, that I chose the limits for the y axis after looking at the graph.
plt.legend()                   # Improving your plot is often an iterative process!
plt.show()
