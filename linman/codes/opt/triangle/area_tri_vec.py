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



#Line points
A = np.array([1,1,1]) 
B = np.array([1,2,3]) 
C = np.array([2,3,1]) 

a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)

#Hero's formula
s = (a+b+c)/2
print(s,a,b,c)
area = np.sqrt(s*(s-a)*(s-b)*(s-c))
print("Hero's formula", area)

area = 1/2*np.linalg.norm(np.cross(B-A,C-A))
print("Cross Product", area)
