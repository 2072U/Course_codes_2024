import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = x*e^(-2.0*x)
def function(x):
    return x * np.exp(-2.0 * x)

# Define the exact solution of the given differential equation 
def exact_solution(x):
    return -np.exp(-2.0 * x) / 4 - x * np.exp(-2.0 * x) / 4 + np.exp(-2) * x / 2.0

# Function to create the differentiation matrix for the finite difference method
def SetDiffMatrix(n):
    """
    Generates a tridiagonal matrix representing the finite difference approximation of the second derivative operator.
    Parameters:
    - n: int, number of interior mesh points
    Returns:
    - A: numpy.ndarray, tridiagonal matrix
    """
    A = np.zeros((n, n))  # Initialize the matrix
    for i in range(n):
        A[i, i] = -2.0  # Fill the diagonal
    for i in range(n - 1):  # Fill up the super and sub diagonals
        A[i + 1, i] = 1.0
        A[i, i + 1] = 1.0
    return A

# Domain
left = 0.0
right = 1.0

# Boundary conditions
alpha = -0.25     #u(0) = alpha
beta = 0.0        #u(-1)= beta

# Grid for plotting
number_plot_points = 1000
x_plot = np.linspace(left, right, number_plot_points)

#Start and end indices 
iStart = 3
iEnd = 10
iTest = iEnd - iStart + 1

# Initialize an array to store errors
errors = np.zeros((iTest, 2))

# Loop over different mesh sizes
for i in range(iStart, iEnd + 1):
    n = 2**i

    # Interior mesh points
    xj = np.linspace(left, right, n + 2)

    # Width of the subintervals: h = 1 / (n + 1)
    h = (right - left) / float(n + 1)

    #The right-hand side of the differential equation: y = h^2 * f
    y = h**2 * function(xj[1:n + 1])

    # Add boundary conditions to the RHS
    y[0] += alpha   #j=1 additional alpha
    y[n-1] += beta  #j=N additional beta

    # Tridiagonal matrix A
    A = SetDiffMatrix(n)

    # Calculate the condition number of A
    cond = np.linalg.cond(A)
    
    #Print the condition number for n
    print('Condition number of A is %f for n=%d.' % (cond, n))

    # Solve the system to find the approximate solution u
    approx_solution_u = np.linalg.solve(-A, y)

    # Apply boundary conditions to the approximate solution
    approx_solution_u = np.insert(approx_solution_u, 0, alpha)  # u(0) = alpha
    approx_solution_u = np.append(approx_solution_u, beta)  # u(1) = beta

    # Error at each grid point
    error = np.absolute(approx_solution_u - exact_solution(xj))
    errors[i - iStart, :] = [n, np.max(error)]

    # Plot exact and approximate solutions
    plt.plot(xj, approx_solution_u, '*r', label='Approximate')
    plt.plot(x_plot, exact_solution(x_plot), '-k', label='Exact')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('n=' + str(n))
    plt.show()

# Plot actual error and order(h^2)
plt.loglog(errors[:, 0], errors[:, 1], '-*', label='Actual Error')
plt.loglog(errors[:, 0], 2e-2 * errors[:, 0]**(-2), '-r', label='Order h^2')
plt.xlabel('Number of Mesh Points')
plt.ylabel('Error of Numerical Solution')
plt.legend()
plt.show()

