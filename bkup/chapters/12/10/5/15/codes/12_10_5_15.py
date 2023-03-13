import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/nithish/Downloads/training-main/math/codes/CoordGeo/')
from line.funcs import *


a = np.array([1,2])
b = np.array([-2,1])

line_OA = line_gen(np.array([0,0]),a) 
line_OB = line_gen(np.array([0,0]),b)

print('a:\n',a)
print('b:\n',b)

print('norm of vector a: ',np.linalg.norm(a))
print('norm of vector b: ',np.linalg.norm(b))

print('a+b:\n',a+b)

print('(a+b).T (a+b): ',(a+b).T @ (a+b))
print('norm of vector a squared + norm of vector b squared: ',np.linalg.norm(a)**2+np.linalg.norm(b)**2)


plt.scatter(a[0],a[1])
plt.scatter(b[0],b[1])
plt.annotate('A(1,2)',(a[0],a[1]))
plt.annotate('B(-2,1)',(b[0],b[1]))

plt.plot(line_OA[0,:],line_OA[1,:])
plt.plot(line_OB[0,:],line_OB[1,:])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axis('equal')
plt.grid(visible='True',axis='both')
plt.savefig("/home/nithish/Documents/EE5610_Pattern_Recognition_Machine_Learning/quiz6/12.10.5.15/figs/vector_plot.jpg")
plt.show()


