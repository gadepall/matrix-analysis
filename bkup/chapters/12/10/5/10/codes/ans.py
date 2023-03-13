import numpy as np

a = np.array([2,-4, 5])
b = np.array([1, -2, -3])

#Diagonal d1
d1 = (a+b)
print(d1)

#Diagonal d2
d2 = (a-b)
print(d2)

#unit vector along d1
d1_unit = d1/np.linalg.norm(d1)
print(d1_unit)

#unit vector along d2
d2_unit = d2/np.linalg.norm(d2)
print(d2_unit)

#Area of parallelogram
area = np.linalg.norm(np.cross(a,b))
print(area)