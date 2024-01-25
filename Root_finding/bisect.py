# Bisection for finding the root of a function. L. van Veen, Ontario Tech U, 2024.
import numpy as np

# Inputs: function handle f, initial domain (a,b), max nr. of iterations kMax, tolrance for error (epsX) and residual (epsF).
def bisect(f, a, b, kMax, epsX, epsF):
    record = np.zeros((kMax,3))                    # This serves only to plot the errors/residuals later.
    conv = 0                                       # Convergence flag set to "no convergence" by default.
    l = a
    r = b
    for i in range(0,kMax):                        # Loop iver iterations.
        m = (l+r)/2                                # Find the mid point.
        fm = f(m)                                  # Compute the function value there and at the left boundary.
        fl = f(l)
        err = abs(r-l)                             # Compute the error and residual.
        res = abs(f(m))
        record[i,:] = [i,err,res]
        print('It %d, x=%f, err=%e' % (i,m,err))
        if err < epsX and res < epsF:              # Test for convergence.
            conv = 1
            break
        if fl * fm > 0:                            # Update the domain.
            l = m
        else:
            r = m

    if conv == 0:                                  # Print a warning if the there was no convergence.
        print('No convergence in bisection.')
    return m,record                                # You may want to return the convergence flag, too!
