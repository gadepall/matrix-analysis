import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *

import sys, os                                          #for path to external scripts
sys.path.insert(0,'/home/chirag/matrix/CoordGeo') 


#local imports
#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c

#Input parameters
r=3/2
O=np.array(([0,0]))

V_1=np.array([[0,0],[0,4]]) 

#for parabola
V = np.array([[1,0],[0,0]])
u = np.array(([0,-2]))
f = 0

#for computing stright line
lamda,P = LA.eigh(V_1)
print(P)
print("lamda is",lamda)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda) # e value 
    P = np.flip(P,axis=1)   #e vectors in col
print(P)

#for parabola
lamda,P = LA.eigh(V) 
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda) # e value 
    P = np.flip(P,axis=1)   #e vectors in col

#
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.5]))
fl = LA.norm(F)


#Finding k values for Points A and B
m = np.array([1,0]) #direction vector

d = np.sqrt((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
#print(k1)
#print(k2)
A = F + k1*m
B = F + k2*m

print("intersection of Point A " ,A)
print("intersection of Point B" ,B)

#pmeters to generate parabola
num_points = 1700
delta = 20*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


##Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

x_circ= circ_gen(O,r)

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:])


#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

# To find area 
a1=np.where(x_circ[1,:] >= 1/2)[0]
a2=np.where(p[1,:] <= 1/2) [0]


cir_st=a1[0]
cir_ed=a1[-1]+1

p_st=a2[0]
p_ed=a2[-1]+1

y_1=(x_circ[1, cir_st:cir_ed])
y_2=(p[1, p_st:p_ed])

x_1 = (x_circ[0, cir_st:cir_ed])
x_2 = (p[0, p_st:p_ed])

area_1=np.trapz(y_1,x_1)
area2=np.trapz(x_2,y_2)

area1=abs(area_1)

print("Area of circle alone the above points",area1)
print("Area of parabola alone the above points is",area2)
print("The area of the portion is",area1-area2)



#Shading the region 
#ind = int(len)
#fill_betweenx(x_circ[1,0:ind],p_x[0:ind],x_circ[0,0:ind],facecolor='orange')
#plt.fill_between(np.array([-np.sqrt(2), np.sqrt(2)]), y1=1.5, y2=0)

       
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
#plt.savefig(os.path.join(script_dir, fig_relative))
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()
