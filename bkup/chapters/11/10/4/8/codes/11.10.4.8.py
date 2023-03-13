import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys

#range of x
x=np.linspace(-1,5,10)
#1st line equation
y=x
#2nd line equation
y1=-x
#k value for 3rd line equation
k=int(input('enter the vakue of k='))

#plot the 1st line equation -x+y=0
plt.plot(x,y,'r',label='y-x=0')

#plot the 2nd line equation x+y=0
plt.plot(x,y1,'b',label='x+y=0')

#plot the 3rd line equation x-k=0
plt.axvline(x=k,color='g',label='x-k=0')

#intersection of two lines
def line_intersect(n1,c1,n2,c2):
  n=np.vstack((n1.T,n2.T))
  p = np.array([[c1],[c2]])
  #intersection
  p=np.linalg.inv(n)@p
  return p

n1 = np.array([[-1],[1]])
n2 = np.array([[1],[1]])
n3 = np.array([[1],[0]])
c1 = 0
c2 = 0
c3 = k


#Intersection points
A = line_intersect(n1,c1,n2,c2).T
B = line_intersect(n2,c2,n3,c3).T
C = line_intersect(n3,c3,n1,c1).T


#Area of triangle using formula arc(triangleABC)=(1/2)|(A-B)x(A-C)|
area_of_triangle=(1/2)*np.cross((A-B),(A-C))
print('Area of triangle=',area_of_triangle)



#Labelling the coordinates
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels= ['A(0,0)','B(k,-k)','C(k,k)',]
for i, txt in enumerate(vert_labels):
 plt.annotate(txt, #this is text
     (tri_coords[0,i], tri_coords[1,i]), #this is the point to label
    textcoords="offset points" , # How to position the text
    xytext=(0,10),#Distance from the text to points (x,y)
    ha='center') # horizontal alignment can be left , right or center

plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/11.10.4.8/figs/fig.pdf')
plt.show()
