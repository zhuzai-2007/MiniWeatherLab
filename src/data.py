import numpy as np

N=96
SHIFT=12

def fields(n, seed):
    rng=np.random.default_rng(seed)
    x=np.arange(N)[None,None,:]; y=np.arange(N)[None,:,None]
    cx=rng.uniform(20,76,(n,1,1)); cy=rng.uniform(20,76,(n,1,1))
    amp=rng.uniform(6,14,(n,1,1)); sigma=rng.uniform(3.5,6,(n,1,1))
    dx=(x-cx+N/2)%N-N/2; dy=(y-cy+N/2)%N-N/2
    return (-amp*np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))).astype('float32')

def branches(x):
    return np.roll(x,-SHIFT,axis=-1),np.roll(x,SHIFT,axis=-1)
