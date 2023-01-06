#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import matplotlib.cm as cm
import matplotlib.legend as Legend

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
from line.funcs import *
from triangle.funcs import *
from params import *

#if using termux
import subprocess
import shlex
#end if


#Generating points on a standard hyperbola 
def hyperbola_gen(y, a, b):
    x = np.sqrt(1+(y**2)/(b**2))*a
    return x

#Input parameters
V = np.array([[9,0],[0,-16]])
u = np.array(([0,0]))
f = -144 

e1 = np.array(([1,0]))
#Input parameters
a=4
b=3
O=np.array([0,0])

#Vertices of hyperbola
G = np.array([a,0])
H = np.array([-a,0])

I = np.array([0,b])
J = np.array([0,-b])

lamda,P = LA.eigh(V)
if(lamda[0] <= 0):      # If eigen value negative, present at start of lamda 
   lamda = np.flip(lamda)
   P = np.flip(P,axis=1)
e = np.sqrt(1- lamda[0]/lamda[1])
e_square = e**2
print("Eccentricity of hyperbola is ", e)

n = np.sqrt(abs(lamda[1]))*P[:,0]


F1 = (e * np.sqrt((lamda[1])/(-f*(1-e_square)))) * (-f)/(lamda[1]) * e1
F2 = -(e * np.sqrt((lamda[1])/(-f*(1-e_square)))) * (-f)/(lamda[1]) * e1

fl1 = LA.norm(F1)
fl2 = LA.norm(F2)

m = omat@n
#Points for hyperbola Major Axis
ellipAxis_A = 2*n
ellipAxis_B = -2*n

#Points for hyperbola Minor Axis
ellipMinorAxis_A = 1.333*m 
ellipMinorAxis_B = -1.333*m

#points for Latus rectum
lr1_Ay = np.sqrt((-f-lamda[0]*fl1**2)/lamda[1])
A = F1 + np.array([0, lr1_Ay])
B = F1 + np.array([0, -lr1_Ay])

lr2_Ay = np.sqrt((-f-lamda[0]*fl2**2)/lamda[1])
C = F2 + np.array([0, lr2_Ay])
D = F2 + np.array([0, -lr2_Ay])

#Generating the hyperbola
y = np.linspace(-5,5,400)
x1 =  hyperbola_gen(y,a,b) 
x2 =  -hyperbola_gen(y,a,b) 

# Generating lines 
x_AB = line_gen(ellipAxis_A, ellipAxis_B)
x_minor_AB = line_gen(ellipMinorAxis_A, ellipMinorAxis_B)
x_lr1_AB = line_gen(A , B )
x_lr2_CD = line_gen(C , D )

#Plotting the hyperbola
plt.plot(x1,y,label='$Hyperbola$', color = 'blue')
plt.plot(x2,y,label='$Hyperbola$', color = 'blue')

leg_label = "{} {}".format("Transverse", "Axis")
plt.plot(x_AB[0,:],x_AB[1,:] ,label = leg_label)

leg_label = "{} {}".format("Conjugate", "Axis")
plt.plot(x_minor_AB[0,:],x_minor_AB[1,:] ,label = leg_label)

leg_label = "{} {}".format("Latus", "Rectum1")
plt.plot(x_lr1_AB[0,:],x_lr1_AB[1,:] ,label = leg_label)

leg_label = "{} {}".format("Latus", "Rectum2")
plt.plot(x_lr2_CD[0,:],x_lr2_CD[1,:] ,label = leg_label)


#Labeling the coordinates
plot_coords = np.vstack((F1,F2,H,G)).T
vert_labels = ['$F_1$','$F_2$','$V_1$','$V_2$']
for i, txt in enumerate(vert_labels):
    if ( i == 0) :
      label = "{}".format('$F_1 - Focus 1$' )
    elif ( i == 1) :
      label = "{}".format('$F_2 - Focus 2$' )
    elif ( i == 2) :
      label = "{}".format('$V_1 - Vertex 1$' )
    else :
      label = "{}".format('$V_2 - Vertex 2$' )

    plt.scatter(plot_coords[0,i], plot_coords[1,i], s=15, label = label)
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.gca().legend(loc='lower left', prop={'size':6},bbox_to_anchor=(0.91,0.4))
plt.grid() # minor

plt.axis('equal')
plt.title('Hyperbola')
#if using termux
plt.savefig('../figs/problem1.pdf')
#else
plt.show()
