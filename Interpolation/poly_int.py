import numpy as np

def forwardSub(L,y):
    # Solve the triangular system L z = y, where L is lower triangular.
    n = np.shape(L)[0]
    z = np.zeros((n,1))
    for i in range(1,n+1):
        z[i-1] = y[i-1]
        for j in range(1,i):
            z[i-1] -= L[i-1,j-1] * z[j-1]
        z[i-1] /= L[i-1,i-1]
    return z

def phi(x,xs,k):
    q = 1.0
    if k>0:
        for i in range(k):
            q *= (x-xs[i])
    return q

def poly_int(xs,ys,xout):
    n = np.size(xs) - 1
    V = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(i+1):
            V[i,j] = phi(xs[i],xs,j)
    a = forwardSub(V,np.reshape(np.asarray(ys),(n+1,1)))
    yout = np.copy(xout)
    m = np.size(yout)
    for i in range(m):
        yout[i] = 0.0
        for j in range(n+1):
            yout[i] += a[j] * phi(xout[i],xs,j)
    
    return yout

    
