# Simple gradient descent. By L. van Veen, Ontario Tech U, 2024.
# Lecture 23 of MATH/CSCI2072U, 2024.
import numpy as np

# Inputs: initial guess z, cost function f(z), gradient of f, max nr. of iterations, threshold for gain/step size.
def GD(z0,f,gradf,maxIt,eps):
    conv = False                                 # Set convergence flag to False by default.
    z = np.copy(z0)                              # Local variable for variables, z.
    res = f(z)                                   # Cost function at the initial guess.
    ss = 0.01                                    # Initial step size.
    beta = 0.8                                   # Parameter to decide in increasing/decreasing step size.
    gamma = 2.0                                  # Factor by which we increase/decrease step size.
    errs = [[0,res,0,ss]]                        # Order: it nr., cost function, gain, step size.
    for i in range(maxIt):                       # Loop over iterations.
        v = gradf(z)                             # Find the gradient (direction of steepest increase).
        normGrad = np.linalg.norm(v,2)           # Find the norm.
        z -= ss * v / normGrad                   # Take step of size ss agains the gradient.
        new_res = f(z)                           # Re-compute the cost function.
        gain = (res - new_res) / res             # The gain is the relative decrease of the cost function.
        if gain > beta * ss * normGrad / res:    # Compare the gain to that for a linear function. If the gain
            ss *= gamma                          # is at least beta times the linear gain, speed up.
        elif gain < ss * normGrad / (beta * res):# However, if it is smaller than the linear gain/beta, slow down.
            ss /= gamma
            
        res = np.copy(new_res)
        errs.append([i,res,gain,ss])
        print('Iteration %d, res=%e, gain=%e.' % (i,new_res,gain))
        if gain < eps or ss < eps:               # Check for convergence (small gain or step size), exit if done.
            conv = True
            break
    return z,np.asarray(errs)
        
