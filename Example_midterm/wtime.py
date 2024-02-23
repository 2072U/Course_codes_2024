# wtime: measures the wall time taken by a call to list_primes as a function of the input n.
# Part of the example midterm discussed in lecture 12.
import numpy as np
import matplotlib.pyplot as plt
from list_primes import *
from time import time

kstart = 3
kend = 10
wtimes = []
for k in range(kstart,kend+1):
    n = 2**k                                          # Choosing values like this gives evenly spaced points on the log-log scale.
    E = time()                                        # Measure current time.
    L = list_primes(n)                                # Call our own function for listing primes.
    E = time() - E                                    # Measure the elapsed time.
    wtimes.append([n,E])
print(wtimes)                                         # Visial inspection: do our numbers make sense?
wtimes = np.asarray(wtimes)
plt.loglog(wtimes[:,0],wtimes[:,1],'-*')
plt.loglog([8,1024],[1e-7* 8**3,1e-7* 1024**3],'-r')
plt.xlabel('N')
plt.ylabel('wall time (s)')
plt.title('straight line indicates O(N^3)')
plt.show()
