import numpy as np

veca=np.array([1,1,1])
vecb=np.array([2,-1,3])
vecc=np.array([1,-2,1])
u=2*veca-vecb+3*vecc
magu=np.linalg.norm(u)
uv=u/magu
print(uv)
