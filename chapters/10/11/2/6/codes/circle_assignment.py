#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys, os                                          #for path to external scripts
script_dir = os.path.dirname(__file__)
lib_relative = '../../../CoordGeo'
fig_relative = '../figs/fig1.pdf'
sys.path.insert(0,os.path.join(script_dir, lib_relative))

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
a = 6   #AB = 6
b = 8   #BC = 8
A = np.array(([0,a]))
B = np.array(([0,0]))
C = np.array(([b,0]))
#O = (B+C)/2
m = A - C
D = A + (((m.T)@(B-A))/LA.norm(m)**2)*m
O = 0.5*LA.inv(np.vstack((B-C, C-D)))@np.array([LA.norm(B)**2-LA.norm(C)**2, LA.norm(C)**2-LA.norm(D)**2]) 
#At = A - O  #Point A when origin shifted to O
At = np.array(([8,0]))
r = b/2
e1 = np.array([1,0])
e2 = np.array([0,1])
a = (e1.T@At)**2 + (e2.T@At)**2
b = -2*(r**2)*(e1.T@At)
c = (r**2)*(r**2 - (e2.T@At)**2)
t1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
t2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
Ep = np.array([t1, np.sqrt(r**2 - t1**2)])
En = np.array([t1, -np.sqrt(r**2 - t1**2)])
if((At-Ep).T@Ep == 0):
    E = Ep + O
else:
    E = En + O
Bp = np.array([t2, np.sqrt(r**2 - t2**2)])
Bn = np.array([t2, -np.sqrt(r**2 - t2**2)])
if((At-Bn).T@Bn == 0):
    B = Bn + O
else:
    B = Bp + O
print(t1)
print(t2)
print(E)
print(B)
'''

##Generating all shapes
x_AB = line_gen(A,B)
x_BC = line_gen(C,B)
x_CA = line_gen(A,C)
x_BD = line_gen(B,D)
x_AE = line_gen(A,E)
c_rO = circ_gen(O,r)

#Plotting all shapes
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])
plt.plot(x_BD[0,:],x_BD[1,:])
plt.plot(x_AE[0,:],x_AE[1,:])
plt.plot(c_rO[0,:],c_rO[1,:])


#Labeling the coordinates
plot_coords = np.vstack((A,B,C,D,E,O)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A','B','C','D','E','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig(os.path.join(script_dir, fig_relative))
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
#plt.show()
'''
