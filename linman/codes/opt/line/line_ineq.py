#Code by GVV Sharma
#December 22, 2019
#released under GNU GPL
#Line Inequality

import matplotlib.pyplot as plt
from coeffs import *

#if using termux
import subprocess
import shlex
#end if

affine = np.array(([1,1],[-1,1]))
c =  np.array([5,-3])

#Original axes
points = np.array([[0, 0], [0, 1],[1, 0]])

#Transformed axes
affine_points = np.linalg.inv(affine)@(c+points).T
affine_points = affine_points.T

#Filling up the desired region
plt.fill(affine_points[:,0], affine_points[:,1], 'k', alpha=0.3)

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_ineq.pdf')
plt.savefig('./line/figs/line_ineq.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_ineq.pdf"))
#else
#plt.show()

	

