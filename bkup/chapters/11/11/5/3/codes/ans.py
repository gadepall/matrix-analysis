import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import sympy as sp

import sys   
sys.path.insert(0,'/home/lokesh/EE2802/EE2802-Machine_learning/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from params import *

def parab_gen(x, a):
    y = (x**2)/(4*a)
    return y

def parab_gen_conic(x):
    lambda_2 = sp.Symbol('lamda_2')
    X = np.array([x, lambda_2])
    eq = sp.simplify(np.transpose(X)@V@X + 2*np.transpose(u)@X + f)
    y = sp.solve(eq, 'lamda_2')
    y = np.array(list(y.items()))
    return y[0][1]



#representative figure consruction
x = np.linspace(-50, 50, 200)
a = 20 #random value of A for representative figure
y = parab_gen(x, a)

O = np.array([0, 0])
A = np.array([-50, parab_gen(-50, a)])
B = np.array([50, parab_gen(50, a)])

plt.plot(x,y,label='$Parabola$', color = 'blue')

#Labeling the coordinates
plot_coords = np.vstack((O,A,B)).T
vert_labels = ['$O$','$A$','$B$']
for i, txt in enumerate(vert_labels):
    plt.scatter(plot_coords[0,i], plot_coords[1,i], s=15)
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center

ground_lvl1 = np.array([-50, -6])
ground_lvl2 = np.array([50, -6])
ground = line_gen(ground_lvl1, ground_lvl2)

centre_pt1 = np.array([0, -10])
centre_pt2 = np.array([0, 35])
centre_line = line_gen(centre_pt1, centre_pt2)

plt.plot(ground[0,:],ground[1,:] ,'r', label = 'ground')
plt.plot(centre_line[0,:], centre_line[1, :], 'g--' )
#name the ground line as l on plot
plt.gca().legend(loc='lower left', prop={'size':6},bbox_to_anchor=(0.91,0.4))
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.11.5.3/figs/1.png')

#clear the plot
plt.clf()


#Find the u and f for parabola equation
V = np.array([[1, 0], [0, 0]])

O = np.array([0, 0])
A = np.array([50, 24])
B = np.array([-50, 24])

#find u and f
matrix_3 = -np.array([[np.transpose(O)@V@O], [np.transpose(A)@V@A], [np.transpose(B)@V@B]])
matrix_1 = np.array([[2*np.transpose(O)[0], 2*np.transpose(O)[1], 1], [2*np.transpose(A)[0], 2*np.transpose(A)[1], 1], [2*np.transpose(B)[0], 2*np.transpose(B)[1], 1]])

matrix_2 = np.round(np.linalg.solve(matrix_1, matrix_3),2) #rounding off the matrix entries to 2 decimal places

u = np.array([matrix_2[0], matrix_2[1]])
f = matrix_2[2]

lambda_1 = sp.Symbol('lamda_1')
lambda_2 = sp.Symbol('lamda_2')

X = np.array([lambda_1, lambda_2])
eq = sp.simplify(np.transpose(X)@V@X + 2*np.transpose(u)@X + f)

x = np.arange(-50,50,1)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = parab_gen_conic(x[i])

#given that lamda_1 = 18
D = np.array([18, parab_gen_conic(18)])


plt.plot(x,y,label='$Parabola$')
plt.grid()
plt.plot(ground[0,:],ground[1,:], 'r' ,label = 'ground')
plt.plot(centre_line[0,:], centre_line[1, :], 'g--' )

plot_coords = np.vstack((O,A,B,D)).T
vert_labels = ['$O$','$A$','$B$', '$D$']
for i, txt in enumerate(vert_labels):
    plt.scatter(plot_coords[0,i], plot_coords[1,i], s=15)
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center
plt.gca().legend(loc='lower left', prop={'size':6},bbox_to_anchor=(0.91,0.4))
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.11.5.3/figs/parabola.png')