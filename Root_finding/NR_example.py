# Example of a system of nonlinear equations, Lecture 11, MATH/CSCI2072U, by L. van Veen, Ontario Tech, 2024.
import numpy as np
from NR import NR
import matplotlib.pyplot as plt

# The task is to maximize the profit Q = s u - p c (nr of items sold time salec price minus number produced times prodction cost).
# We take u and p to be our variables, (u,p)=(x[0],x[1]), take c to be constant and model the sales s.
# Example parameters:
uc = 4.0   # Market price of one item.
pc = 30.0  # Market capacity.
c = 1.0    # Production cost per unit.

# Define the problem. Let x=(u,p) anf f=(dQ/du, dQ/dp). We modelled s as a decreasing function of u and p.
# The maximum is found by setting dQ/du=0=dQ/dp.
def f(x):
    f = np.zeros((2,1))
    f[0] = (x[1]/(x[1]/pc+1.0)) * np.exp(-x[0]/uc) * (1.0 - x[0]/uc)
    f[1] = x[0] * np.exp(-x[0]/uc) /(x[1]/pc+1.0) - x[0] * x[1] * np.exp(-x[0]/uc) / (pc * (x[1]+1)**2) - c
    return f
# The Jacobian of these equations is necessary for Newton-Raphson iteration. Note, that J[1,0]=J[0,1]=d^2 f/ dudp.
def J(x):
    Jac = np.zeros((2,2))
    Jac[0,0] = -np.exp(-x[0]/uc) * x[1] * pc * (2.0 * uc - x[0]) / (uc**2 * (x[1] + pc))
    Jac[0,1] = np.exp(-x[0]/uc) * pc**2 * (uc - x[0]) / (uc * (x[1] + pc)**2)
    Jac[1,0] = np.exp(-x[0]/uc) * pc**2 * (uc - x[0]) / (uc * (x[1] + pc)**2)
    Jac[1,1] = -np.exp(-x[0]/uc)*x[0]*((pc**2 - 1.0)*x[1]**3 + (3*pc**2 - 2*pc + 1.0) * x[1]**2 + (2*pc**2 + 2*pc)*x[1] + 2*pc**2)/((x[1] + pc)**2*pc*(x[1] + 1)**3)
    return Jac

# Parameters of Newton-Raphson iteration:
epsf = 1E-10
epsx = 1E-10
itmx = 10
# Initial point:
x0 = np.array([[uc],[pc]])

# Call our own Newton-Raphson function:
x, err, res, conv = NR(f,J,x0,epsx,epsf,itmx)

print(x)
if conv == 1:
    print("Note the quadratic convergence!")
its = np.shape(err)[0]

# Plot the residuals and errors, note that the convergence is faster than linear:
plt.figure
plt.semilogy(range(0,its),err[0:its,0],'-*k')
plt.semilogy(range(0,its),res[0:its,0],'-*r')
plt.xlabel('Nr. of iterations')
plt.ylabel('residual (red) and error (black)')
plt.title('Convergence for 2 x 2 test problem')
plt.show()
