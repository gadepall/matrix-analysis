
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')

#local imports
from conics.funcs import circ_gen
from line.funcs import line_gen

#if using termux
import subprocess
import shlex
#end if

I = np.array(([1,0],[0,1]))
#Given points for the circle
Q = np.array(([0,0]))  # External Point
O = np.array(([25,0])) # Centre
r = 7  #Radius of the circle
u = -O
f = 576

sigma = (np.outer(Q+u, Q+u) - (LA.norm(Q)**2 + 2*u.T@Q+f)*I )

lamda, P = LA.eigh(sigma)
if(lamda[0] <= 0):      # If eigen value negative, present at start of lamda 
   lamda = np.flip(lamda)
   P = np.flip(P,axis=1)

lamda_vec1 = np.array(([np.sqrt(abs(lamda[0])), np.sqrt(abs(lamda[1]))]))
lamda_vec2 = np.array(([np.sqrt(abs(lamda[0])), -np.sqrt(abs(lamda[1]))]))

#Normals
n1 = P@lamda_vec1
n2 = P@lamda_vec2

#Tangent Points
R11 = r*(n1/LA.norm(n1)) - u
R21 = -r*(n1/LA.norm(n1)) - u
R12 = r*(n2/LA.norm(n2)) - u
R22 = -r*(n2/LA.norm(n2)) - u

##Generating the circle
x_circ= circ_gen(O,r)

#Generating lines
x_cR21 = line_gen(Q, R21)
x_cR22 = line_gen(Q, R22)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_cR21[0,:],x_cR21[1,:],label='$Tangent_1$')
plt.plot(x_cR22[0,:],x_cR22[1,:],label='$Tangent_2$')


#Labeling the coordinates
plot_coords = np.vstack((O,Q,R22,R21)).T
vert_labels = ['O','Q','$R_1$','$R_2$']
for i, txt in enumerate(vert_labels):
    if ( i == 0) :
      label = "{}".format('$O - Center$' )
    elif ( i == 3) :
      label = "{} {}".format('$R_2 - Tangent$', '$Point 2$' )
    elif ( i == 2) :
      label = "{} {}".format('$R_1 - Tangent$', '$Point 1$' )
    else :
      label = "{} {}".format('$Q - External$', '$Point$' )

    plt.scatter(plot_coords[0,i], plot_coords[1,i], s=15, label = label)
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.gca().legend(loc='best', prop={'size':6})
plt.grid() # minor
plt.axis('equal')
plt.title('Circle')

#if using termux
plt.savefig('../figs/problem1.pdf')
#else
plt.show()
