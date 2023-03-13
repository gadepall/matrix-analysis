import numpy as np

# Define the vector
veca = np.array([1,2,-3])
vecb = np.array([-1,-2,1])
m=np.subtract(vecb,veca)
# magnitude of direction vector
mag = np.linalg.norm(m)
cos1=m[0]/mag
cos2 =m[1]/mag
cos3=m[2]/mag
print(cos1,cos2,cos3)

