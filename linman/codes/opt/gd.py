#Gradient Descent
#By GVV Sharma, October 25, 2018
#Algo from Wikipedia
#Revised  by Aayush Arora
#Jan 13, 2019
#Released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shlex
from coeffs import *

def f(x,a,b,d):
	return a*(x**2)+b*x+d

#Line parameters
n =  np.array([3,-4]) 
c = 26
P = np.array([3,-5]) 

A,B = line_icepts(n,c)
m = omat@n
#Parabola parameters
a = np.linalg.norm(m)**2
b = 2*m.T@(A-P)
d = np.linalg.norm(A-P)**2

#Gradient Descent
cur_x = 2 # The algorithm starts at x=1
gamma = 0.001 # step size multiplier
precision = 0.00000001
previous_step_size = 1 
max_iters = 100000000 # maximum number of iterations
iters = 0 #iteration counter

df = lambda x: 2*a*x + b

while (previous_step_size > precision) & (iters < max_iters):
    prev_x = cur_x
    cur_x -= gamma * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)
    iters+=1
print(f(cur_x,a,b,d))
print("The local minimum occurs at", cur_x)
print("The minimum distance is", np.sqrt(f(cur_x,a,b,d)))
print("The theoretical distance is", np.abs(n@P-c)/np.linalg.norm(n))
