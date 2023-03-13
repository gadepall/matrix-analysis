import sys                                                    #for path to external scripts
sys.path.insert(0,'/home/sinkona/Documents/CoordGeo')         #path to my scripts

from line.funcs import *

#To find the area of a quadrilateral

#import module
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import subprocess
import shlex

##Finding area using Matrices
A=np.array(([-4,5,1]))
B=np.array(([0,7,1]))
C=np.array(([5,-5,1]))
D=np.array(([-4,-2,1]))

x = np.block([[A],[B],[C]])
det1 = 0.5*np.linalg.det(x)  

y = np.block([[A],[D],[C]])
det2 = 0.5*np.linalg.det(y)  

print("Area of triangle ABC =",-float(det1))
print("Area of triangle ADC =",float(det2))
print("Area of quadrilateral ABCD =",-float(det1)+float(det2))

##Finding area using Cross product
A=np.array(([-4,5]))
B=np.array(([0,7]))
C=np.array(([5,-5]))
D=np.array(([-4,-2]))

v1=B-A
v2=B-C
ar_t1=0.5*np.linalg.norm((np.cross(v1,v2)))

v3=D-A
v4=D-C
ar_t2=0.5*np.linalg.norm((np.cross(v3,v4)))

print("Area of triangle ABC =",ar_t1)
print("Area of triangle ADC =",ar_t2)
print("Area of quadrilateral ABCD) =",ar_t1+ar_t2)

##Quadrilateral ABCD
# initialize x and y coordinates
x1 = [-4, 0, 5, -4, -4]
y1 = [ 5, 7,-5, -2,  5]
pts1=['D','C','B','A']
plt.scatter(x1,y1)

plt.plot(x1,y1)
vert_labels = ['A(-4,5)','B(0,7)','C(5,-5)','D(-4,-2)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (x1[i], y1[i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis([-8,8,-8,10])
plt.show()

plt.savefig('/sdcard/Download/fwc-main/Assignment1/line/line1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/fwc-main/Assignment1/line/line1.pdf'"))    

##Quadrilateral ABCD with diagonal AC    
x1 = [-4, 0, 5, -4, -4, 5]
y1 = [ 5, 7,-5, -2,  5,-5]
pts1=['D','C','B','A']
plt.scatter(x1,y1)

plt.plot(x1,y1)
vert_labels = ['A(-4,5)','B(0,7)','C(5,-5)','D(-4,-2)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (x1[i], y1[i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center    
    
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis([-8,8,-8,10])
plt.show()
