# By L van Veen, Ontario Tech U, 2023.
# Steffenson iteration for assignment 2.
import numpy as np

def steff(f,fp,x,epse,epsr,itmx):
# Steffensen iteration for f(x)=0.
#   Input: f, f', intial point x, error tolerance, residual tolerance, max nr of iterations
#   Output: approximate solution x, estimate of error, residual
    conv=0                                             # Flag for convergence.
    xs = np.zeros((3))                                 # We need to remember three intermediary values so we pre-allocate an array.
    for i in range(0,itmx):                            # Loop over iterations.
        xs[0] = x                                      # First point is x.
        dx = -f(x)/fp(x)                               # First Newton step.
        x = x+dx                                       
        xs[1]=x;                                       # Second point after one Newton step.
        dx = -f(x)/fp(x)                               # Second Newton step
        x = x+dx
        xs[2] = x                                      # Third point after second Newton step.
        x=xs[0]-(xs[1]-xs[0])**2/(xs[2]-2.0*xs[1]+xs[0]) # Steffensen's formula.
        err=abs(x-xs[0])                               # Difference between previous and current point is an estimate of the error.
        res=abs(f(x))                            # Residual.
        print("Iteration "+str(i)+" x="+str(x)+" err="+str(err)+" res="+str(res))
        if err < epse and res < epsr:                  # Check for convergence.
            print("Converged, exiting...")
            conv = 1
            break
    if conv == 0:
        print("No convergence!")
    return x,err,res


