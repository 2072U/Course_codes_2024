import numpy as np

def Newton(f,fp,x0,kMax,epsX,epsF):
    record = np.zeros((kMax,3))
    x = x0
    conv = 0
    for i in range(kMax):
        r = f(x)
        df = fp(x)
        dx = -r/df
        err = abs(dx)
        res = abs(r)
        record[i,:] = [i,err,res]
        print('Iteration %d, err=%e, res=%e' % (i,err,res))
        if err < epsX and res < epsF:
            conv = 1
            break
        x += dx
    if conv == 0:
        print('No convergence!')
    return x,record
