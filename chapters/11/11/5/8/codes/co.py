import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys, os                                          #for path to external scripts
sys.path.insert(0, '/sdcard/Download/chinna/matix/CoordGeo')
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import parab_gen

#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c
#Input parameters
i= 1
V = np.array([[0,0],[0,1]])
u = np.array(([-2*i,0]))
f = 0
theta1=((np.arccos(1/2))+(np.arccos(1)))/2
theta2=((np.arccos(1))-(np.arccos(1/2)))/2
print(theta1)
h=4*i*np.cos(theta1)
j=np.sin(theta1)**2
r=h/j
print(h)
print(j)
print(r)
lamda,P = LA.eigh(V) 
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda)       
    P = np.flip(P,axis=1)
eta = u@P[:,0] 
a = np.vstack((u.T + eta*P[:,0].T, V))     
b = np.hstack((-f, eta*P[:,0]-u))
center = LA.lstsq(a,b,rcond=None)[0]
O=center
n = np.sqrt(lamda[1])*P[:,0]
#n = np.array(([0,1]))
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)      
F = (c*n - u)/lamda[1]
fl = LA.norm(F)   
x=12*i
O=np.array(([0,0]))
A=np.array(([r*np.cos(theta1),r*np.sin(theta1)]))
B=np.array(([r*np.cos(theta2),r*np.sin(theta2)]))
print(A)    
print(O)    
print(B)
print(LA.norm(A-O))
print(LA.norm(B-A))
print(LA.norm(B-O))
num_points =50
delta = 2*np.abs(fl)/10
p_y = np.linspace(-10*np.abs(fl)-delta,10*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


##Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T

# Generating lines after transforming points
x_AB = line_gen(A,B)
x_AO = line_gen(A,O)
x_BO = line_gen(B,O)

#Plotting all shapes
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_AO[0,:],x_AO[1,:])
plt.plot(x_BO[0,:],x_BO[1,:])
plt.plot(p[0,:], p[1,:])


#Labeling the coordinates
plot_coords = np.vstack((A,B,O)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A','B','O']
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
plt.savefig('/sdcard/Download/chinna/matrix/conics_assignment/co.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/Download/chinna/matrix/conics_assignment/co.pdf'"))
