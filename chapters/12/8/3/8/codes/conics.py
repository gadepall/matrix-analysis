
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/DCIM/Assignment-4/CoordGeo')         #path to my scripts

#Circle Assignment
import subprocess
import shlex
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *


#Input parameters
a=3
b=2
O=np.array([0,0])
x=np.arange(0,3.1,0.1) #range of values to shade the region


#Generating the ellipse
elli=  ellipse_gen(a,b) 

#Points of intersection of a conic section with a line
m=np.array([1/b,-1/a]);
q=np.array([a,0]);
V=np.block([[b**2,0],[0,a**2]]);
u=np.array([0,0]);
f=-((a**2)*(b**2));
p1,p2=inter_pt(m,q,V,u,f)

#Generating  line
x_R1 = line_gen(p1,p2);

#shading the region
y1=(b/a)*(np.sqrt(a**2-x**2))
y2=(b/a)*(a-x)

#finding the area of smaller region bounded by the ellipse and the line
#area of triangle
def integrand1(x, a, b):
    return ((b/a)*(a-x))
A1,err=quad(integrand1, 0, a, args=(a,b))
#area of ellipse
def integrand1(x, a, b):
    return ((b/a)*(np.sqrt(a**2-x**2)))
A2,err=quad(integrand1, 0, a, args=(a,b))
area=A2-A1;
print('the area of smaller region bounded by the ellipse and the line is ',area);



# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#Plotting all line
plt.plot(x_R1[0,:],x_R1[1,:],label='$line$')


#Plotting the ellipse
plt.plot(elli[0,:],elli[1,:],label='$ellipse$')



#Labeling the coordinates
tri_coords = np.vstack((p1,p2,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O']
for i, txt in enumerate(vert_labels):
       plt.annotate(txt,					 # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", 		# how to position the text
                 xytext=(0,10),				 # distance from text to points (x,y)
                 ha='center') 				# horizontal alignment can be left, right or center


plt.fill_between(x,y1,y2,color='green', alpha=.2)
plt.legend()
plt.grid(True) # minor
plt.axis('equal')
#plt.savefig('conic_fig.png')
#plt.show()
plt.savefig('/sdcard/DCIM/Assignment-6/codes/conic_fig.pdf')
subprocess.run(shlex.split("termux-open sdcard/DCIM/Assignment-6/codes/conic_fig.pdf"))
