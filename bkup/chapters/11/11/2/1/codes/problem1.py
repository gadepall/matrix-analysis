#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')

#local imports
#from CoordGeo.line.funcs import *
#from triangle.funcs import *
from conics.funcs import parab_gen
from params import *
#if using termux
import subprocess
import shlex
#end if

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

def affine_transform(P,c,x):
    return P@x + c

#Input parameters
V = np.array([[0,0],[0,1]])
u = np.array(([-6,0]))
f = 0

lamda,P = LA.eigh(V)
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = (c*n - u)/lamda[1]
fl = LA.norm(F)
m = omat@n
d = np.sqrt((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
A = F + k1*m
B = F + k2*m

num_points = 50
delta = 2*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-2*delta,2*np.abs(fl)+2*delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y

#Points for Directrix
C = np.array(([c,0]))
direct_A = 8*m+C  
direct_B = -8*m+C

#Points for Parabola Axis
parAxis_A = 6*n
parAxis_B = -6*n

##Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T

# Generating lines after transforming points
x_AB = line_gen(A,B)
x_direct_AB = line_gen(direct_A, direct_B)
x_parAxis_AB = line_gen(parAxis_A, parAxis_B)

#Plotting all shapes
leg_label = "{} {}".format("Latus", "Rectum")
plt.plot(x_AB[0,:],x_AB[1,:] ,label = leg_label)
plt.plot(x_direct_AB[0,:],x_direct_AB[1,:], label ='$Directrix$')
leg_label = "{} {}".format("Parabola", "Axis")
plt.plot(x_parAxis_AB[0,:],x_parAxis_AB[1,:],label= leg_label)
plt.plot(p[0,:], p[1,:],label ='$Parabola$')


#Labeling the coordinates
plot_coords = np.vstack((A,B,F,O)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A','B','F','O']
for i, txt in enumerate(vert_labels):
    label = "{}({:.0f},{:.0f})".format(txt, plot_coords[0,i],plot_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
                 (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(18,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

label = "{}{:.0f}".format("x = ", c) #Form label as A(x,y)
plt.annotate(label, # this is the text
             (plot_coords[0,2]-8, plot_coords[1,2]+2), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(18,5), # distance from text to points (x,y)
             rotation = 90,
             ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best', fontsize = 'small')
plt.grid() # minor
plt.axis('equal')
plt.title('Parabola')
#if using termux
plt.savefig('../figs/problem1.pdf')
#subprocess.run(shlex.split("termux-open '../figs/problem1.pdf'")) 
#plt.savefig(d$os.path.join(script_dir, fig_relative))
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
#plt.show()
