#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import matplotlib.cm as cm
import matplotlib.legend as Legend

import sys   
sys.path.insert(0,'/home/lokesh/EE2802/EE2802-Machine_learning/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from params import *

#Generating points on a standard hyperbola 
def hyperbola_gen(x, a, b):
    y = np.sqrt(1+(x**2)/(b**2))*a
    return y

#Input parameters
V = np.array([[-1,0],[0,5/9]])
u = np.array(([0,0]))
f = -4

e2 = np.array(([0,1]))
#Input parameters
a=6/np.sqrt(5)
b=2
O=np.array([0,0])

#Vertices of hyperbola
G = np.array([0,a])
H = np.array([0,-a])

I = np.array([b,0])
J = np.array([-b,0])

lamda,P = LA.eigh(V)
if(lamda[0] <= 0):      # If eigen value negative, present at start of lamda 
   lamda = np.flip(lamda)
   P = np.flip(P,axis=1)
e = np.sqrt(1- lamda[0]/lamda[1])
e_square = e**2
print("Eccentricity of hyperbola is ", e)

n = np.sqrt(abs(lamda[1]))*P[:,0]

F1 = (e * np.sqrt((lamda[1])/(-f*(1-e_square)))) * (-f)/(lamda[1]) * e2
F2 = -(e * np.sqrt((lamda[1])/(-f*(1-e_square)))) * (-f)/(lamda[1]) * e2

fl1 = LA.norm(F1)
fl2 = LA.norm(F2)

#Points for hyperbola Major Axis
ellipAxis_A = np.array([5,0])
ellipAxis_B = np.array([-5,0])

#Points for hyperbola Minor Axis
ellipMinorAxis_A = np.array([0,5])
ellipMinorAxis_B = np.array([0,-5])

#points for Latus rectum
lr1_Ay = np.sqrt((-f-lamda[0]*fl1**2)/lamda[1])
A = F1 + np.array([lr1_Ay,0])
B = F1 + np.array([-lr1_Ay,0])

lr2_Ay = np.sqrt((-f-lamda[0]*fl2**2)/lamda[1])
C = F2 + np.array([lr2_Ay,0])
D = F2 + np.array([-lr2_Ay,0])

#Generating the hyperbola
x = np.linspace(-4,4,100)
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
plt.title('Hyperbola')
print(f"Focii are : {F1,F2}")
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.11.4.5/figs/hyperbola.png')