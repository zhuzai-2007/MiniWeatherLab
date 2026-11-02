"""Practical data and metric invariants for the paired experiment."""
import numpy as np
from data import fields,branches,N,SHIFT
parts=[fields(n,s) for n,s in ((1024,11),(128,22),(128,33))]
# IDs of the generating RNGs are disjoint; check realized fields for accidental equality.
all_fields=np.concatenate(parts)
features=all_fields[:,::12,::12].reshape(len(all_fields),-1)
assert len(np.unique(features,axis=0))==len(all_fields)
for x in parts:
    l,r=branches(x)
    assert np.array_equal(np.roll(l,SHIFT,axis=-1),x)
    assert np.array_equal(np.roll(r,-SHIFT,axis=-1),x)
    mean=(l+r)/2
    oracle=(np.mean((mean-l)**2)+np.mean((mean-r)**2))/2
    blind=np.mean((l-r)**2)/2
    assert np.isclose(oracle*2,blind,rtol=1e-6)
print('PASS: disjoint sampled fields; exact periodic shifts; oracle MSE is half the blind-direction MSE.')
