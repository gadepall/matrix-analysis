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
d1=4
B1 = d1*np.array(([np.cos(theta1),np.sin(theta1)]))
x_AB1=line_gen(A,B1)

plt.plot(x_AB1[0,:],x_AB1[1,:],label='d1={}'.format(d1))
sqr_vert = np.vstack((A,B1)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['(0,0)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
y = (1/ma.sqrt(3))*x+8/(ma.sqrt(3))
plt.plot(x, y, '-r', label='y=x/sqrt(3)+8/sqrt(3)')

plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()                                      
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line1.png')
subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line1.png"))

plt.show()
