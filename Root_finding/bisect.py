import numpy as np

def bisect(f, a, b, kMax, epsX, epsF):
    record = np.zeros((kMax,3))
    conv = 0
    l = a
    r = b
    for i in range(0,kMax):
        m = (l+r)/2
        fm = f(m)
        fl = f(l)
        err = abs(r-l)
        res = abs(f(m))
        record[i,:] = [i,err,res]
        print('It %d, x=%f, err=%e' % (i,m,err))
        if err < epsX and res < epsF:
            conv = 1
            break
        if fl * fm > 0:
            l = m
        else:
            r = m

    if conv == 0:
        print('No convergence in bisection.')
    return m,record
