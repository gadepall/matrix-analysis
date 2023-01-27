import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
 
 

A = np.array(([2,3])) 
B = np.array(([4,5]))
C = A+B
D = A-B
E = 2*np.cross(A,B)
F = np.cross(D,C)
if(E.all()==F.all()):
  print("(A-B)X(A+B)=2(AXB)")
else:
  print("your assumption is wrong")
