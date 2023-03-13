#Python libraries for math and graphics
import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad
import mpmath as mp

#import sys                                          #for path to external scripts
#sys.path.insert(0,'/home/susi/CoordGeo')
#Generating points on a circle
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ
#Generating points on a parabola
def parab_gen(y,a):
	x = y**2/a
	return x

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

#Points of intersection of a conic section with a line
def inter_pt(m,q,V,u,f):
    a = m@V@m
    b = m@(V@q+u)
    c = conic_quad(q,V,u,f)
    l1,l2 =np.roots([a,2*b,c]) 
#    print(a,b,c)
    x1 = q+l1*m
    x2 = q+l2*m
    return x1,x2 

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#generating circle
#Input parameters
p = 4
#r2 = 2*r1
O = np.zeros(2)

theta = mp.pi/2
A = p*(np.array(([mp.cos(theta/3),mp.sin(theta/3)])))
B = p*(np.array(([-mp.cos(theta/3),mp.sin(theta/3)])))
#print(A)
#print(B)

##Generating all lines
x_circ1 = circ_gen(O,p)


##Plotting all lines
plt.plot(x_circ1[0,:],x_circ1[1,:],label='$circle1$')
#  plotting parabola
simlen=100
a=6
y= np.linspace(-5,5,simlen)
x = (y**2)/6
plt.plot(x, y, label='Parabola')
#Labeling the coordinates
tri_coords = np.vstack(O)
plt.scatter(tri_coords[0,:], tri_coords[1,:])   
vert_labels = ['O']
for i,  txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[0,]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#Points of intersection of a conic section with a line
m=np.array(([0,1]))
V=np.array(([0,0],[0,1]))
u=np.array(([-3,0]))
f=0

q=np.array(([2,0]))
p1,p2=inter_pt(m,q,V,u,f)
print(p1)
print(p2)

#area of circle
A1=np.pi*(p**2)
print("Area of circle is",A1)
def integrand1(x):
   return (np.sqrt(6*x))
a1,err=quad(integrand1, 0, 2)
def integrand1(x):
   return ((np.sqrt(16-(x**2))))
a2,err=quad(integrand1, 2, 4)
A2=(2*(a1+a2))
print("area interior to the circle and parabola",A2)
A3=A1-A2
print("the area of the circle exterior to the parabola",A3)

#if using termux
plt.savefig('/sdcard/Download/codes/conics/cp.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/codes/conics/cp.pdf"))
#plt.show()
