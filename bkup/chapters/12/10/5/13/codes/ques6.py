import numpy as np
a=np.array([1,1,1])
b=np.array([2,4,-5])
c=np.array([1,2,3])
x=b+c
magnitude=np.linalg.norm(x)
u=x/magnitude
result=a@u
print(result)



