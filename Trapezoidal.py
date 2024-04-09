import numpy as np

def Trapezoidal(function, left, right, n_subintervals):
    width_subintervals = (right - left) / n_subintervals
    x_intervals = left + np.linspace(0, n_subintervals, n_subintervals + 1) * width_subintervals

    area = 0.0
    for i in range(n_subintervals - 1):
        area += function(x_intervals[i + 1])

    area = width_subintervals * (area + (function(x_intervals[0]) + function(x_intervals[-1])) / 2.0)

    return area
