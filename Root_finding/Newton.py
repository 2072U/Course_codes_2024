# Newton iteration for finding the root of a function. L. van Veen, Ontario Tech U, 2024.
import numpy as np

# Inputs: function handles f and fp (derivative of f), initial guess x0, max nr. of iterations kMax, tolrance for error (epsX) and residual (epsF).
def Newton(f,fp,x0,kMax,epsX,epsF):
    record = np.zeros((kMax,3))                    # This serves only to plot the errors/residuals later.
    x = x0                                         # Convergence flag set to "no convergence" by default.
    conv = 0
    for i in range(kMax):                          # Loop over iterations.
        r = f(x)                                   # Evaluate f...
        df = fp(x)                                 # ... and its derivative.
        dx = -r/df                                 # Compute the Newton update step.
        err = abs(dx)                              # Compute an estimate of the error and the residual.
        res = abs(r)
        record[i,:] = [i,err,res]
        print('Iteration %d, err=%e, res=%e' % (i,err,res))
        if err < epsX and res < epsF:              # Check for comvergence.
            conv = 1
            break
        x += dx                                    # Update the approximate solution.
    if conv == 0:
        print('No convergence!')
    return x,record                                # You may want to return the convergence flag, too!
