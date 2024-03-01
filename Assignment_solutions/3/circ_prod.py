# Example solution to assignmet A3Q3, MATH/CSCI2072U, Ontario Tech U, 2024, by L. van Veen.
# Multiplication of a banded, circulant matrix with a vector.
import numpy as np

# In: arrays of n floats a, b and c representing the nonzero elements of the matrix and x, representing the vector to multiply.
# Out: matrix-vector product, another array of n floats.
def circ_prod(a,b,c,x):
    n = np.shape(a)[0]                                          # Extract the matrix size.
    y = np.zeros((n,1))                                         # Allocate an array for the result.
    y[0] = a[0] * x[0] + c[0] * x[1] + b[0] * x[n-1]            # Copmute the first and last element of the product separately.
    y[n-1] = c[n-1] * x[0] + b[n-1] * x[n-2] + a[n-1] * x[n-1]
    for i in range(1,n-1):                                      # Loop over rows 2 through n-1.
        y[i] = b[i] * x[i-1] + a[i] * x[i] + c[i] * x[i+1]

    return y

