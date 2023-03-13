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
P=np.array([1,5])
X=np.array([4,3])

#Direction vector
m=X-P
n=np.array([2,3])
z=np.matmul(n,m)

x_PX=line_gen(P,X)
plt.plot(x_PX[0,:],x_PX[1,:],label='$PX$')
sqr_vert = np.vstack((P,X)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['P(1,5)','X(4,3)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
y= ((-2/3)*x+4)
plt.plot(x, y, '-r', label=('[2,3]X=-4'))
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper right')
plt.grid()                                      
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/11.10.3.11/figs/fig.png')
plt.show()
