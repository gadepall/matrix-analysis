import numpy as np

m1 = np.array([3,-16,7])
m2 = np.array([3,8,-5])

m = np.array([2,3,6])

mat = np.array([m1,m2])

print('m1:\n',m1)
print('m2:\n',m2)
print('mat:\n',mat)
print('m:\n',m)

print('mat @ m:\n',mat @ m)

print('Hence, m is perpendicular to m1 and m2\n')
