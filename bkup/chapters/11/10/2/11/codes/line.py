#Code by Shreyash Chandra (works on termux)
#October 10, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the equation of line which is perpenducular to line segment joining the points (1,0) & (2,3) divides it in the ratio 1:n . 


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *

#if using termux
import subprocess
import shlex
#end if
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/CoordGeo/CoordGeo')         #path to my scripts
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen


#local imports
#Orthogonal matrix
omat = np.array([[0,1],[-1,0]])

#Rotation Matrix
def rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return  np.array([[c,-s],[s,c]])

def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

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

# given points on coordinates as vectors
A = np.array([1,0])
B = np.array([2,3])

N = norm_vec(A,B)

#direction vector of line joining 2 given points is
M = dir_vec(A,B)

# position/directional vector of the A&B line segment used as the normal vector of its perpendicular line

#The point P that divides the line segment AB in the ratio 1 : n is given by#P= (B + nA)/(1+n)
n  = int(input("enter your required n for 1:n ratio (n>=0) :  ")) 
n = abs(n)
u= (2+n)/(1+n)   
e= (3)/(1+n)   
p = np.array([u,e]) #abtained point P by section formula in 1:n ratio 
#print (p,"point ofint")


#The equation of the line perpendicular to #n⊤ (x) = C and passing through the point P is given by "m⊤ (x − P) = 0"
C = M@p       #Constant C 
print("---------------------------------------------")
print ("Equation of perpendicular line dividing given line-segment in 1 :",n,"ratio is:")
print(M,"X =", C)
print ("-----------------------------------------------")
#############plot#######################
#Input parameters
a = np.array((N,M))

b = (([3,((2+n)/(1+n)+((9/(1+n))))]))

e1 = np.array(([1,0])) # standard basic vector
n1 = a[0,]
n2 = a[1,]
c1 = b[0]
c2 = b[1]

#Direction vectors
m1 = omat@n1
m2 = omat@n2

#Points on the lines
x1 = c1/(n1@e1)
A1 =  x1*e1
x2 = c2/(n2@e1)
A2 =  x2*e1


#Generating all lines
k1=-3
k2=0
x_AB = line_gen(A,B)

x_CD = line_dir_pt(m2,A2,k1,k2)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Given line-segment')
plt.plot(x_CD[0,:],x_CD[1,:],label='{}X={}'.format(M,C) )

#Labeling the coordinates
tri_coords = np.vstack((A,B,p)).T
#tri_coords = p.T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
#plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['P','Q','R']
#plt.plot(x_circ[0,:],x_circ[1,:],label='x^2+y^2−2x+2y=47')
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
plt.title('line divides PQ in ratio 1:{}'.format(n))
#if using termux
plt.savefig('/sdcard/download/python/line/fig/linefig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/download/python/line/fig/linefig.pdf"))
#else
#plt.show()
