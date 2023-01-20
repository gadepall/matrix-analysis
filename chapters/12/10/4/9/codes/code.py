import numpy as np
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *


A = np.array([1,1,2])
B = np.array([2,3,5])
C = np.array([1,5,5])

crossproduct = np.cross(B-A, C-A)
area = 1/2 * np.linalg.norm(crossproduct) 
print(area)
