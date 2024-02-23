# list_primes: for positive integer p list all primes between 1 and p (inclusive).
# Part of the example midterm discussed in lecture 12. By L. van Veen, Ontario Tech U, 2024.
from list_factors import *

def list_primes(n):
    L = []
    for i in range(1,n+1):
        factors = list_factors(i)
        if len(factors) == 1:
            L.append(i)
    return L
