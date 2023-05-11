import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
import math

#line generation function
def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB


#To construct triangles
a = 9
c = 5
theta = np.pi*2/3
A = np.array([0,0])
e1=np.array([1,0])
B = a*e1
C = c*np.array([np.cos(theta),np.sin(theta)])
D= c*np.array([np.cos(-theta),np.sin(theta)])

m_1 = D-A
m_2 = D-B
n_1 = C-A
n_2 = C-B

m1 = np.linalg.norm(m_1)
m2 = np.linalg.norm(m_2)
n1 = np.linalg.norm(n_1)
n2 = np.linalg.norm(n_2)

#Finding angles
angADB =np.degrees(np.arccos(np.dot(m_1,m_2)/(m_1*m_2)))
angACB =np.degrees(np.arccos(np.dot(n_1,n_2)/(n_1*n_2)))
if(round(angADB) == round(angACB)):
    print(" ∠ ADB = ∠ ACB")

#line generation with the calculated distances(d1,d2,d3)
x_AB = line_gen(A,B)
x_BD = line_gen(B,D)
x_DA = line_gen(D,A)
x_CA = line_gen(C,A)
x_BC = line_gen(B,C)
x_DC = line_gen(D,C)



plt.plot(x_AB[0,:],x_AB[1,:],label='$distance(AB)$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$distance(BC)$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$distance(CA)$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$distance(DA)$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$distance(BD)$')
plt.plot(x_DC[0,:],x_DC[1,:],label='$distance(DC)$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


line1 = [[2.92, 0.72], [2.80, 1.12]]  # First line from X to Y
line2 = [[2.92, -0.72], [2.80, -1.12]]  # Second line from Y to Z
plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'r-')  # First line
plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'g-')  # Second line
plt.xlabel('$x$')                    
plt.ylabel('$y$')
plt.legend(['$AD$','$AC$','$AB$','$BC$','$BD$','$DC$'])
plt.grid(True)
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/arduino/Vector/figs/fig.png')

plt.title('Quadrilateral ABCD')
plt.show()
