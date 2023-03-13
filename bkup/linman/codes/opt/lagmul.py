#Code by GVV Sharma, 
#Jan 17, 2020
#Released under GNU GPL
#Lagrange Multipliers
import numpy as np

#Line parameters
n =  np.array([3,-4]).reshape(2,-1)
c = 26
P = np.array([3,-5]).reshape(2,-1) 

#Matrix equation
A = np.block([[np.eye(2),-n],[n.T, 0]])
b = np.block([[P],[c]])
print (A,np.linalg.inv(A)@b)
