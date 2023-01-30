#Code by GVV Sharma
#March 26, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if



#Triangle sides
#2-D matrix
a_mat =  np.array(([11,2],[0,-np.sqrt(2)]))
b_mat =  np.array(([1,11],[1,0]))
den_mat = np.array(([1,2],[1,-np.sqrt(2)]))

a = np.linalg.det(a_mat)/np.linalg.det(den_mat)
b = np.linalg.det(b_mat)/np.linalg.det(den_mat)
c = b

print(a,b,c)








