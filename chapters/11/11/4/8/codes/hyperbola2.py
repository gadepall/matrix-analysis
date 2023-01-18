#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import matplotlib.cm as cm
import matplotlib.legend as Legend

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from params import *

#if using termux
import subprocess
import shlex
#end if


#Generating points on a standard hyperbola 
def hyperbola_gen(x, a, b):
    y = np.sqrt(1+(x**2)/(b**2))*a
    return y

#Input parameters
f = -975
a=5
b=sqrt(39)
O=np.array([0,0])

#Vertices of hyperbola
G = np.array([0,a])
H = np.array([0,-a])

I = np.array([b,0])
J = np.array([-b,0])


F1 = np.array([0,8])
F2 = np.array([0,-8])

fl1 = LA.norm(F1)
fl2 = LA.norm(F2)

#Points for hyperbola Major Axis
ellipAxis_A = np.array([0,10])
ellipAxis_B = np.array([0,-10])

#Points for hyperbola Minor Axis
ellipMinorAxis_A = np.array([10,0])
ellipMinorAxis_B = np.array([-10,0])

#points for Latus rectum
A = F1 + np.array([7.8,0])
B = F1 + np.array([-7.8,0])


C = F2 + np.array([7.8,0])
D = F2 + np.array([-7.8,0])


#Generating the hyperbola
x = np.linspace(-10,10,400)
y1 =  hyperbola_gen(x,a,b) 
y2 =  -hyperbola_gen(x,a,b) 

# Generating lines 
x_AB = line_gen(ellipAxis_A, ellipAxis_B)
x_minor_AB = line_gen(ellipMinorAxis_A, ellipMinorAxis_B)
x_lr1_AB = line_gen(A , B )
x_lr2_CD = line_gen(C , D )

#Plotting the hyperbola
plt.plot(x,y1,label='$Hyperbola$', color = 'blue')
plt.plot(x,y2,label='$Hyperbola$', color = 'blue')

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
plt.savefig('/sdcard/Download/latexfiles/conics/figs/hyperbola2.png')
plt.show()
