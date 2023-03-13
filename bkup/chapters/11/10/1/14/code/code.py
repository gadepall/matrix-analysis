import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *

A = np.array([1985, 92])
B = np.array([1995, 97])

x_AB = line_gen(A,B)
plt.plot(x_AB[0,:],x_AB[1,:])

tri_coords = np.vstack((A,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(1985,92)','B(1995,97)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(10,-20), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlim(1980, 2005)
plt.ylim(88,100)
plt.xlabel("Years")
plt.ylabel("Population in Crores")
plt.grid()
plt.savefig("Fig.png")
plt.show()

# Direction vector of the line joining A, B
m = B-A
print(m)

# Slope
slope = m[1]/m[0]
print(slope)

# Normal vecor
n = omat@m
print(n)

# constant c
c = n@A
print (c)

# Basis vectors
e1 = np.array([1,0])
e2 = np.array([0,1])

# the value of y coordinate for x= 2010
x= 2010
y = (c - (x* (e1@n)))/(e2@n)

print(y)
