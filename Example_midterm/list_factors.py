# list_factors.py: for positive integer n return a list of factorizations. Part of the exampke midterm test discussed in lecture 12.
# By L. van Veen, Ontario Tech U, 2024.

def list_factors(n):
    L = []
    for i in range(1,n+1):
        for j in range(i,n+1):   # range(1,n+10 is also correct but you must then add an if-statement and add the pair only if j>=i
            p = i * j
            if p == n:
                L.append([i,j])
    return L
