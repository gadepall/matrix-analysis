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
theta3 = np.pi*7/4
d3=2*(ma.sqrt(2))
B3 = d3*np.array(([np.cos(theta3),np.sin(theta3)]))
x_AB3=line_gen(A,B3)
plt.plot(x_AB3[0,:],x_AB3[1,:],label='d3={}'.format(d3))
sqr_vert = np.vstack((A,B3)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['(0,0)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
x = np.linspace(-5,5,100)
y = x-4
plt.plot(x, y, '-r', label='y=x-4')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line3.png')
subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/line/11.10.3.3/figs/line3.png"))


plt.show()
