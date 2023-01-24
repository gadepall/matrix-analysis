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
    


x = np.linspace(-5,5,100)
A=np.array(([0,0]))
theta1 = np.pi*2/3
slope = np.tan(theta1)
print(slope)
d1=4
B = d1*np.array(([np.cos(theta1),np.sin(theta1)]))
C= d1*np.array(([-(np.cos(theta1)),-(np.sin(theta1))]))
x_AC=line_gen(A,C)
x_AB=line_gen(A,B)
print(C)

plt.plot(x_AB[0,:],x_AB[1,:],label="Slope = -1.732")
plt.plot(x_AC[0,:],x_AC[1,:],label="theta = 120 degree")
sqr_vert = np.vstack((A,B)).T
sqr_vert1 = np.vstack((A,C)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
plt.scatter(sqr_vert1[0,:],sqr_vert1[1,:])
vert_labels = ['(0,0)']


plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()                                      
plt.axis('equal')
#if using termux
#plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line1.png')
#subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line1.png"))

plt.show()