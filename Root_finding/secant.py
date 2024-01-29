# The secant method for finding the root of a function. F. Majithia and L. van veen, Ontario Tech U, 2024.
import numpy as np

# Inputs: function handle f, two initial points x0 and x1 (floats), max nr of iterations kMax (int), tolerance for the residual and error epsX and epsF (floats). 
def secant_method(f, x0, x1, kMax, epsX, epsF):
    f_x0 = f(x0)                                             # Evaluate the function at the initial points.
    f_x1 = f(x1)
    data_secant = []                                         # Serves only to plot the errors and residuals.
    conv = 0                                                 # No convergence by default, only set to True if the criteria are met.
    for k in range(kMax):                                    # Loop over iterations.
        residual = abs(f_x1)                                 # Compute the residual and error estimate.
        error = abs(x1-x0)
        if residual < epsF or error < epsX:                  # If the residual or estimated error is less than the threshold, exit. In some cases, you may need an AND instead of an OR.
            # If convergence criteria is met:
            conv = 1
            break

        # Secant update:
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        f_x2 = f(x2)
        
        # Record data
        data_secant.append([k, x2, error, residual])

        # Prepare for next iteration (i.e. "forget" one point and add the new one):
        x0, x1 = x1, x2
        f_x0, f_x1 = f_x1, f_x2

    return x1, np.asarray(data_secant)

