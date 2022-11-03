import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys, os
script_dir = os.path.dirname(__file__)
lib_relative = '../../coord'
sys.path.insert(0,os.path.join(script_dir, lib_relative))

#local imports 
from square.funsq import *

##set length of square
a = 10 

[O,A,B,C,M] = sq_vert(a)

##gen lines
X_OA = line_gen(O,A)
X_BA = line_gen(A,B)
X_BC = line_gen(B,C)
X_CO = line_gen(C,O)
X_D1 = line_gen(O,B)
X_D2 = line_gen(C,A)

#ploting lines
plt.plot(X_OA[0,:], X_OA[1,:])
plt.plot(X_BA[0,:], X_BA[1,:])
plt.plot(X_BC[0,:], X_BC[1,:])
plt.plot(X_CO[0,:], X_CO[1,:])
plt.plot(X_D1[0,:], X_D1[1,:])
plt.plot(X_D2[0,:], X_D2[1,:])

#diagonals
D1 = A+B
D2 = A-B

#length of diagonals
d1 = LA.norm(D1)
d2 = LA.norm(D2)

#midpoint of diagonals
M1 = (D1/2)
M2 = (D2/2)

#angle between diagonals

dot_product = D1 @ D2
# if dot product is 0 then vectors are perpendicular

#unit_D1 = D1 / LA. norm(D1)
#unit_D2 = D2 / LA. norm(D2)
#dot_product = (unit_D1 @ unit_D2)
#angle = np.arccos(dot_product)
#angle_degree = math.degrees(angle)

#print statements
print('vertices of the square are')
print('O=',O, ' A=',A,' B=',B,' C=',C)
print('length of diagonals are')
print( 'D1=',d1,'D2=',d2)
print('dot product of diagonals = ', dot_product)
#print(vector_1,vector_2, unit_vector_1, unit_vector_2, dot_product)
print('Midpoint of the diagonal D1=', M1, 'D2=',M2)
print('For a square, Length of diagonals are equal, they are perpendicular, and bisect eachother')

#graph 
sq_coords = np.vstack((A,B,C,O,M)).T
plt.scatter(sq_coords[0,:], sq_coords[1,:])
vert_labels = ['A','B','C','O','M']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (sq_coords[0,i], sq_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or centre
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('../figs/sq_plot.png')
#plt.show()

