import matplotlib.pyplot as plt
import numpy as np
import math as ma
from matplotlib import pyplot as plt, patches
import math
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB
    

#Input parameters
x = np.linspace(-5,5,100)
theta = 0
A=np.array([np.cos(theta),np.sin(theta)])
phi=np.pi/2
B=np.array([np.cos(phi),np.sin(phi)])
x_AB=line_gen(A,B)
d=np.sin(np.pi/2)/(2*(np.sin(np.pi/4)))
P=d*np.array([np.cos(np.pi/4),np.sin(np.pi/4)])
O=np.array([0,0])
x_PO=line_gen(P,O)

#plotting lines
plt.plot(x_AB[0,:],x_AB[1,:],label='(1 1)x=1')
plt.plot(x_PO[0,:],x_PO[1,:],label='$PO$')
#Labeling the coordinates
sqr_vert = np.vstack((A,B,P,O)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['A','B','P','O(0,0)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')

plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper right')  
plt.grid()                                      
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/Download/codes/lines/11.10.4.5/figs/fig.png')

plt.show()
