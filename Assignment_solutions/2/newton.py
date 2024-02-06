# Author: L. van Veen, UOIT, 2018.
# Function for Newton iteration.
# Inputs: function handles F and its derivative DF, intial guess x0, tolerance for error tolx and residual tolf, maximal number of iterations.
# Outputs: final approximation x, estimate of final error err and final residual res.
# Error message is printed if the tolerances are not obtained.

def newton(F,DF,x0,tolf,tolx,kmax):
    x = x0                                                # Initialize the approximate solution.
    conv = 0                                              # By default set the convergence flag to 0 (false).
    for k in range(1,kmax):                               # Loop over Newton iterations.
        r = F(x)                                          # Compute residual.
        dx = -r/DF(x)                                     # Compute update step.
        x = x + dx                                        # Apply update step.
        err = abs(dx)                                     # Estimate error.
        res = abs(F(x))                                   # Compute residual.
        print('it=%d x=%e err=%e res=%e' % (k,x,err,res)) # Print error and residual for diagnostic purposes.
        if err < tolx and res < tolf:                     # Check for convergence.
            conv = 1                                      # Flag convergence.
            break                                         # Exit loop.
    if conv == 0:                                         # if no convergence after kmax iterations print warning.
        print('No convergence!')
    return x,err,res
