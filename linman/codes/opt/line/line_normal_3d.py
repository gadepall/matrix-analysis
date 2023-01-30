#Code by GVV Sharma
#November 12, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

#if using termux
import subprocess
import shlex
#end if



#Plane points
R = np.array([2,5,-3]) 
S = np.array([-2,-3,5]) 
T = np.array([5,3,-3]) 
print(np.cross(R-S,S-T))

