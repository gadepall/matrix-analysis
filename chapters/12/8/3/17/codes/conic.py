import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/susi/Documents/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

import subprocess
import shlex
#  plotting parabola
t=np.arange(-1,1,0.01)
t1 = (t ** 2)
t2 = (-t**2)
plt.plot(t, t1, label='Parabola')
plt.plot(t, t2, label='Parabola')
A=np.array([1,0])
B=np.array([1,1])
O=np.array([0,0])
D=np.array([-1,-1])
C=np.array([-1,0])
# Shaded region
#Fill under the curve
plt.fill_between(
        x= t, 
        y1= t1, 
        where= (0 < t)&(t < 1),
        color= "b",
        alpha= 0.2)
        
#Fill under the curve
plt.fill_between(
        x= t, 
        y1= t2, 
        where= (-1< t)&(t < 0),
        color= "b",
        alpha= 0.2)
#print for Area
x=sym.Symbol('x',real=True)
f=x**2
z=f.subs(x,0)     #substitute
ie=sym.integrate(f,(x,-1,0))
print("Area of OCD = ",ie)
print(" = ",float(ie))
z=f.subs(x,1)     #substitute
#Area of OAB
f=x**2
ie1=sym.integrate(f,(x,0,1))
print("Area of OAB = ",ie1)
print(" = ",float(ie1))

#Required Area
s=(ie+ie1)
print("Required area = ",s)
print(" = ",float(s))
t_cor1 = [0,0]
t1_cor1 = [-2, 2]
plt.plot(t_cor1, t1_cor1, 'r')
t_cor1 = [-2, 2]
t1_cor1 = [0, 0]
plt.plot(t_cor1, t1_cor1, 'r')
t_cor1 = [1, 1]
t1_cor1 = [5, -5]
plt.plot(t_cor1, t1_cor1, 'b--')
t_cor1 = [-1, -1]
t1_cor1 = [5, -5]
plt.plot(t_cor1, t1_cor1, 'b--')
plt.annotate('X=1',  # this is the text
                 ([1, -5]), # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(-1, 5),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center
plt.annotate('X=-1',  # this is the text
                 ([-1, -5]), # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(6, 5),  # distance from text to points (x,y)
                 ha='center') 
                 
tri_coords = np.vstack((A,B,C,D,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B(1,1)','C','D(-1,-1)','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') #
plt.axis('equal')
plt.legend(loc='best')
plt.grid()
plt.show()
#if using termux
#plt.savefig('/home/susi/Documents/conicfig.pdf')
#subprocess.run(shlex.split("termux-open '/home/susi/Documents/conicfig.pdf'")) 
#else
