import numpy as np
arr1 = [1, -7, 7]
arr2 =  [3, -2, 2]
print("Vector1........\n", arr1)
print("Vector2........\n", arr2)
p=np.cross(arr1,arr2)
x=0
y=19
z=19
r=np.sqrt((x**2)+(y**2)+(z**2))
print(r)
magnitude=np.linalg.norm(p)
print(magnitude)
