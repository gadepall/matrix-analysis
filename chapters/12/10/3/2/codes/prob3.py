
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
 
 

A = np.array(([1,-2,3])) 
B = np.array(([3,-2,1]))
x= (A.T)@B
y=(np.linalg.norm(A))*(np.linalg.norm(B))
c=x/y
p=math.acos(c)
w=p*(180/math.pi)
print(w)
