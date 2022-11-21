#Find the angle between the x-axis and the line joining the points(3,-1) and (4,-2)

import matplotlib.pyplot as plt
import numpy as np
import math

A=np.array([3,-1])
B=np.array([4,-2])
c=A-B
print(c)
x=np.array([1,0])

D=np.linalg.norm(c)
E=np.linalg.norm(x)

'''g=np.dot(c,x)

h=D*E
cos0=np.divide(g,h)
theta=np.arccos(cos0)      #cos0=a.b/|a||b|
th1=np.degrees(theta) 
print(th1)'''     
  

#l=np.array([-1,1]).reshape(2,1)
#n=l*x
n=c@x
k=D*E
p=n/k
theta=np.arccos(p)*(180/3.14)  
#p=np.dot(l/D,x/E)*(180/3.14)
#th1=math.degrees(p)      #cos0=a@b/||a||.||b||
print(theta)
#plt.plot(A,B)
#plt.show()



#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/root1/Downloads/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if


I =  np.eye(2)
e1 =  I[:,0]
#Input parameters

#Circle parameters
r = 2
b=0
#B = np.zeros(2)
#e2 = np.array(([1,0]))
B =  np.array(([0,2]))
O =  np.array(([2,0])) #normal vector
C =  np.array(([4,0]))
Q =  np.array(([3,-1])) #normal vector
P =  np.array(([4,-2]))
#A=np.array((]))
m = e1
k1 = -10
k2 = 10
xline = line_dir_pt(m,O,k1,k2)
xAB = line_gen(B,O)
xOB = line_gen(O,C)
xOA = line_gen(O,Q)
xOC = line_gen(Q,P)


##Generating the circle
#x_circ= circ_gen(O,r)

#Plotting all lines
#plt.plot(xline[0,:],xline[1,:])
plt.plot(xAB[0,:],xAB[1,:])
plt.plot(xOA[0,:],xOA[1,:])
plt.plot(xOB[0,:],xOB[1,:])
plt.plot(xOC[0,:],xOC[1,:])
#Plotting the circle
#plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')


#Labeling the coordinates
tri_coords = np.vstack((C,B,Q,O,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['C','B','Q','O','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-20.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-20.pdf"))
#else
plt.show()





