#Code by GVV Sharma
#October 22, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *
#if using termux
import subprocess
import shlex
#end if

#Triangle sides
a = 6
c = 5
B = np.pi*60/180
b = np.sqrt(a**2+c**2-2*a*c*np.cos(B))
print (b)
