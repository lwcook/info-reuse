import numpy as np

def hm_estimator(vqsamp, targ=None):

    vq = np.sort(vqsamp, kind='mergesort')

    if targ is not None:
        ft = lambda h: targ(h)
    else:
        ft = lambda h: custom_target(h, 0, 0, 5)

    M = len(vqsamp)
    vh = [(1./M)*(0.5 + j) for j in range(M)]
    vt = np.array([float(ft(hi)) for hi in vh])
    Dhat = sum([(1./M)*(vq[j] - vt[j])**2 for j in range(len(vq))])

    return float(np.sqrt(Dhat))

def custom_target(x, q0=0, q1=1, p=2):

    if x < 0:
        return q0
    elif x > 1:
        return q1
    else:
        if p > 0:
            return q0 + (q1 - q0)*(x)**p
        else:
            return q1 + (q0 - q1)*(1-x)**abs(p)

def quantile_estimator(vqsamp, quantile=0.9):

    M = len(vqsamp)
    vq = np.sort(vqsamp, kind='mergesort')
    vh = [(1./M)*(j+1) for j in range(M)]
    for ii, (q, h) in enumerate(zip(vq, vh)):
        if h > quantile:
            Dhat = vq[ii]
            break

    return float(Dhat)
