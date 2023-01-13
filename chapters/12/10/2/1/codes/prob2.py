import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
 
 

A = np.array(([1,1,1])) 
B = np.array(([2,-7,3]))
C = np.array(([1/(math.sqrt(3)),1/(math.sqrt(3)),-1/(math.sqrt(3))]))

x = (np.linalg.norm(A))
y = (np.linalg.norm(B))
z = (np.linalg.norm(C))
print("The magnitude of A is ",x)
print("The magnitude of B is ",y)
print("The magnitude of C is ",z)
