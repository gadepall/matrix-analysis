import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sympy import *
import subprocess
import shlex

import sys                                         
sys.path.insert(0,"'/sdcard/Download/c2_fwc/trunk/CoordGeo")   
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *
#for parabola
V = np.array([[1,0],[0,0]])
u = np.array(([0,-2/3]))
f = 0

C = np.array(([0,0]))
r = np.array(([-2,3])) 
s = np.array(([4,12])) 
d = np.array(([-2,0]))
e = np.array(([4,0]))

def affine_transform(P,c,x):
    return P@x + c

#Transformation 
lamda,P = LA.eigh(V)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.5]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 1700
delta = 50*np.abs(fl)/10
p_y = np.linspace(-4*np.abs(fl)-delta,4*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:])

# Conic parameters
V2 = Matrix(([1,0],[0,0]))
u2 = Matrix([0, -2/3])
f2 = Matrix([0])
# Point from which the line join
h2 = Matrix([-2,3])
#GeneratingLine
x_RS = line_gen(r,s) 
x_RD = line_gen(r,d)
x_SE = line_gen(s,e)
plt.plot(x_RS[0,:],x_RS[1,:],label='$Straight Line$')
plt.plot(x_RD[0,:],x_RD[1,:],label='$Intercept$')
plt.plot(x_SE[0,:],x_SE[1,:],label='$Intercept$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
print("the intersection of point A",r)
print("the intersection of point B",s)
# To find area 
a1=np.where(x_RS[1,:] >= 1/2)[0]
a2=np.where(p[1,:] <= 51/10) [0]

RS_st=a1[0]
RS_ed=a1[-1]+1

p_st=a2[0]
p_ed=a2[-1]+1

y_1=(x_RS[1, RS_st:RS_ed])
y_2=(p[1, p_st:p_ed])

x_1 = (x_RS[0, RS_st:RS_ed])
x_2 = (p[0, p_st:p_ed])

area_1=np.trapz(y_1,x_1)
area2=np.trapz(x_2,y_2)

area1=abs(area_1)
print("Area of line line alone",area1)
print("Area of parabola alone",area2)
print("The area of the portion is",area1-area2)


# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#Labeling the coordinates
tri_coords = np.vstack((r,s,d,e)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a(-2,3)','b(4,12)','d(-2,0)','e(4,0)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.legend()
plt.show()
plt.savefig('/sdcard/Download/c2_fwc/trunk/conic_assignment/docs/conic.png')
subprocess.run(shlex.split("termux-open '/sdcard/Download/c2_fwc/trunk/conic_assignment/docs/conic.pdf' ")
