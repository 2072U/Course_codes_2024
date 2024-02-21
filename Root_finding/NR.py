# Newton-Raphson iteration. L. van Veen, Ontario Tech U, 2024.
import numpy as np
from LUP import *   # Uses ourt own LUP decomposition & forward/backward substitution code.

# In: function handle f (vector-valued), function handle Df (array-valued), tolerance for error and residual, max nr. of iterations.
def NR(f,Df,x0,epsx,epsf,itmx):
    x = x0                                                     # Use local variable for the approximate solution.
    conv = 0                                                   # Set convergence flag.
    err=np.ones((itmx,1))                                      # Pre-allocate arrays for errors and residuals.
    res=np.ones((itmx,1))
    for k in range(0,itmx):                                    # Loop over Newton-Raphson iterates.
        r = -f(x)                                              # Compute residual vector.
        res[k] = np.linalg.norm(r,2)                           # Compute residual.
        J = Df(x)                                              # Compute Jacobian.
        # Solve the linear system "J dx = r" for the update step:
        L,U,P,ok = LUP(J)                                      # LUP decompose Jacobian
        if ok==0:
            print('(Nearly) singular Jacobian, exiting!')
            break
        y = ForwardSub(L,P,r)
        dx = BackwardSub(U,y)
        # Done! The preceding 6 lines can be replaced by "dx = np.linalg.solve(J,r)".
        err[k] = np.linalg.norm(dx,2)                          # Estimate of error.
        print("It. %d err=%e res=%e" % (k,err[k,0],res[k,0]))  # Print error and residual.
        if res[k] < epsf and err[k] < epsx:                    # Test for convergence.
            conv = 1
            print("Converged, exiting...")
            break
        x = x + dx                                             # Apply update step.

    if conv==0:
        print("No convergece!")

    return x,err[0:k+1,:],res[0:k+1,:],conv
