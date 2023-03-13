import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/dinesh/conics/CoordGeo')         #path to my scripts
#Circle Assignment
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *
#if using termux
import subprocess
import shlex

def affine_transform(P,c,x):
    return P@x + c
#Input parameters
i= 1
V = np.array([[0,0],[0,1]])
u = np.array(([-0.5,0]))
f = 0
lamda,P = LA.eigh(V) 
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda)       
    P = np.flip(P,axis=1)
eta = u@P[:,0] 
a = np.vstack((u.T + eta*P[:,0].T, V))     
b = np.hstack((-f, eta*P[:,0]-u))
center = LA.lstsq(a,b,rcond=None)[0]
n = np.sqrt(lamda[1])*P[:,0]
#n = np.array(([0,1]))
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)      
F = (c*n - u)/lamda[1]
fl = LA.norm(F)   
O=np.array(([0,0]))
num_points =50
delta = 2*np.abs(fl)/10
p_y = np.linspace(-10*np.abs(fl)-delta,10*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y

#Points of intersection of a conic section with a line
m2=np.array([0,1]);#direction vector
q2= np.array([1,0]);
V=np.array([[0,0],[0,1]]);
u=np.array([-0.5,0]);
f=0;
d = np.sqrt((m2.T@(V@q2 + u))**2 - (q2.T@V@q2 + 2*u.T@q2 + f)*(m2.T@V@m2))
k1 = (d - m2.T@(V@q2 + u))/(m2.T@V@m2)
k2 = (-d - m2.T@(V@q2 + u))/(m2.T@V@m2)
print(k1)
print(k2)
a0 = q2 + k1*m2
a1 = q2 + k2*m2
p1,p2=inter_pt(m2,q2,V,u,f)
m1=np.array([0,1]);
q1=np.array([4,0]);
p3,p4=inter_pt(m1,q1,V,u,f)
d = np.sqrt((m1.T@(V@q1 + u))**2 - (q1.T@V@q1 + 2*u.T@q1 + f)*(m1.T@V@m1))
k3 = (d - m1.T@(V@q1 + u))/(m1.T@V@m1)
k4 = (-d - m1.T@(V@q1 + u))/(m1.T@V@m1)
print(k3)
print(k4)
a3 = q1 + k3*m1
a2 = q1 + k4*m1
print(a0)
print(a1)
print(a3)
print(a2)


#Generating  line
x_R1 = line_gen(p1,p2);
x_R2 = line_gen(p3,p4);


#area 
def integrand1(x, a, b):
    return (np.sqrt(x))
A1,err=quad(integrand1, 0, 4, args=(a,b))
#area of ellipse
#def integrand1(x, a, b):
 #   return ()
A2,err=quad(integrand1, 0, 1, args=(a,b))
area=A2-A1;
print('the area of the region between thr lines x=1 and x=4 in the parabola in first quadrant is ',np.abs(area));



# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#Plotting all line
plt.plot(x_R1[0,:],x_R1[1,:],label='$line$')
plt.plot(x_R2[0,:],x_R2[1,:],label='$line$')

##Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T
##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
#Plotting all shapes
plt.plot(p[0,:], p[1,:])



#Labeling the coordinates
tri_coords = np.vstack((p1,p2,p3,p4,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a1','a0','a2','a3','O']
for i, txt in enumerate(vert_labels):
       plt.annotate(txt,					 # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", 		# how to position the text
                 xytext=(0,10),				 # distance from text to points (x,y)
                 ha='center') 				# horizontal alignment can be left, right or center


#plt.fill_between(x,y1,y2,color='green', alpha=.2)
plt.legend()
plt.grid(True) # minor
plt.axis('equal')
plt.savefig('conics1.png')
plt.show()
plt.savefig('/sdcard/dinesh/conics/conics1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/dinesh/conics/conics1.pdf' "))

