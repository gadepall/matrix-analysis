import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/nithish/Downloads/training-main/math/codes/CoordGeo/')
from line.funcs import *

orig = np.array([0,0])

n = np.array([3, 2])

dir_vec = norm_vec(orig,n)

c1 = 7/3

c2 = -6

c = (c1+c2)/2

FOP_1 = perp_foot(n,c1,orig)
FOP_2 = perp_foot(n,c2,orig)
FOP_3 = perp_foot(n,c,orig)

line_1 = line_dir_pt(dir_vec,FOP_1,-3,3)
line_2 = line_dir_pt(dir_vec,FOP_2,-3,3)
line_3 = line_dir_pt(dir_vec,FOP_3,-3,3)


plt.scatter(FOP_1[0],FOP_1[1])
plt.scatter(FOP_2[0],FOP_2[1])
plt.scatter(FOP_3[0],FOP_3[1])

plt.plot(line_1[0,:],line_1[1,:])
plt.plot(line_2[0,:],line_2[1,:])
plt.plot(line_3[0,:],line_3[1,:])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axis('equal')
plt.legend(['line 1','line 2','equidistant line'])
plt.grid(visible='True',axis='both')
plt.savefig("/home/nithish/Documents/EE5610_Pattern_Recognition_Machine_Learning/quiz7/11.10.4.21/figs/line_plot.jpg")
plt.show()

