import sys                                          #for path to external scripts
sys.path.insert(0,'/home/user/matrix-conics/CoordGeo')         #path to my scripts
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *
from sympy import *

#Input parameters
a=1
V=np.array([[1,0],[0,1]])
u=np.array(([0,0]))
f=-1
n=np.array(([1,0]))
lamda,P = LA.eigh(V)
O=np.array([0,0])
x=np.arange(0,3.1,0.1) #range of values to shade the region


#Generating the circle
x_circ= circ_gen(O,1)
#Points of intersection of a conic section with a line
m=np.array(([0,-1]))
F =np.array(([a/np.sqrt(2),0]))
fl = LA.norm(F)
d = np.sqrt((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
A = F + k1*m
B = F + k2*m
print("area:"+str(0.5*LA.norm(np.cross(A-O, B-O))))


import numpy as np
#Generating  line
x_R1 = line_dir_pt(m,F,k1,k2);



#finding the area of smaller region bounded by the ellipse and the line
#area of triangle
def integrand1(x, a):
    return (a-x)
A1,err=quad(integrand1, 0, a, args=(a))
#area of circle
def integrand1(x, a):
    return np.sqrt(a**2-x**2)
A2,err=quad(integrand1, 0, a, args=(a))
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


#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')



#Labeling the coordinates
tri_coords = np.vstack((A,B,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O']
for i, txt in enumerate(vert_labels):
       plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#shading the region
#y1=np.sqrt(a**2-x**2);
#y2=(a/np.sqrt(2))-x;

#plt.fill_between(x,y1,y2,color='grey', alpha=.5)
plt.legend()
plt.grid(True) # minor
plt.axis('equal')

#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-3.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-3.pdf"))
plt.savefig('conic.png')
plt.show()
Footer

