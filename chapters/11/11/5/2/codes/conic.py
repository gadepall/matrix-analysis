import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys, os                                          #for path to external scripts

sys.path.insert(0, '/sdcard/Download/10/codes/CoordGeo')        #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c

#Input parameters
x1 = 5/2
y1=-10
a1 = x1**2/(4*y1)
print(a1)
V = np.array([[1,0],[0,0]])
u = np.array(([0,-2*a1]))
f = 0
#V = np.array([[0.5,-0.5],[-0.5,0.5]])
#u = np.array(([-7/np.sqrt(2),-5/np.sqrt(2)]))
#f = 13
lamda,P = LA.eigh(V)
#print(P)
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
eta = u@P[:,0]
#print(eta)
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u))  
center = LA.lstsq(a,b,rcond=None)[0]
O=center
n = np.sqrt(lamda[1])*P[:,0]
#n = np.array(([0,1]))
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = (c*n - u)/lamda[1]
print(F.shape)
fl = LA.norm(F)
m = np.array(([-1,0]))
F = np.array(([0,-2]))
print(F.shape)
print((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
print(V)
print(m)
print(u)
d = np.sqrt((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
print(k1,k2)
C = F + k1*m
D = F + k2*m
print(C)
print(D)
#print(LA.norm(C-D))
A = np.array(([5/2,-10]))
B = np.array(([-5/2,-10]))
num_points = 100
delta = 2*np.abs(fl)/10
p_y = np.linspace(-18*np.abs(fl)-delta,18*np.abs(fl)+delta,num_points)

##Generating all shapes
p_x = parab_gen(p_y,a1)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
#A = affine_transform(P,c,A_std)
#B = affine_transform(P,c,B_std)



# Generating lines after transforming points
x_AB = line_gen(A,B)
x_CD = line_gen(C,D)
'''
x_AO = line_gen(A,O)
x_BO = line_gen(B,O)
'''
#Plotting all shapes
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
'''
plt.plot(x_AO[0,:],x_AO[1,:])
plt.plot(x_BO[0,:],x_BO[1,:])
'''
plt.plot(p[0,:], p[1,:])


#Labeling the coordinates
plot_coords = np.vstack((A,B,C,D,O)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A(5/2,-10)','B(-5/2,-10)','C(x,-2)','D(-x,-2)','(0,0)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,15), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')


#if using termux
plt.savefig('/sdcard/Download/FWC/trunk/conic_aassignment/fig.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/FWC/trunk/conic_aassignment/fig.pdf'")) 
#else
