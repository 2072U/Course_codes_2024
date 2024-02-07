# By L. van Veen, Ontario Tech U, 2024.
# LU decomposition without pivoting - MAY BREAK DOWN EVEN IF A IS NON-SINHULAR - see lecture notes.
# Input: n X n matrix A.
# Output: n X n matrices L and U such that LU=A and L and U are triangular.
import numpy as np

def LU(A):
    ok = 1                                # Without pivoting, the decomposition might fail so we use a warning flag.
    small = 1e-12                         # If a pivot is smaller than this in absolute value, raise the flag.
    n = np.shape(A)[0]                    # Extract the number of rows and columns.
    U = np.copy(A)                        # Copy A into U making sure they are separate variable.                    
    L = np.identity(n)                    # Initialize L as identity matrix.
    for j in range(1,n):                  # Loop over columns.
        for i in range(j+1,n+1):          # Loop over elements below the pivot.
            if abs(U[j-1,j-1]) < small:   # Raise error flag and exit if the pivot is too small.
                print("Near-zero pivot!")
                ok = 0
                break
            L[i-1,j-1] = U[i-1,j-1]/U[j-1,j-1]                     # Compute the multiplier.
            for k in range(j,n+1):                                 # Gauss elimination.
                U[i-1,k-1] = U[i-1,k-1] - L[i-1,j-1] * U[j-1,k-1]
    return L,U,ok

