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
A = np.array([1,-1]) 
B = np.array([-4,6]) 
C = np.array([-3,-5]) 

a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)

#Hero's formula
s = (a+b+c)/2
print(s,a,b,c)
area = np.sqrt(s*(s-a)*(s-b)*(s-c))
print("Hero's formula", area)

M=np.vstack((B-A,C-A))
area = 1/2*np.abs(np.linalg.det(M))
print("Determinant  formula", area)
