import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def line_gen(A,B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB

#Rhombus can be constructed using two diagonal lengths;
A = np.array(([3,0]))
B = np.array(([4,5]))
C = np.array(([-1,4]))
#find midpoint
M = (A+C)/2
#coords of Fourth point
D_x = 2*M[0]-B[0] 
D_y = 2*M[1]-B[1]
O = np.array(([(A+C)/2]))

D = np.array(([D_x,D_y]))

#line generation
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_AC = line_gen(A,C)
x_BD = line_gen(B,D)

plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')


rhm_vert = np.vstack((A,B,C,D,O)).T
plt.scatter(rhm_vert[0,:],rhm_vert[1,:])
vert_labels = ['A','B','C','D','O']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (rhm_vert[0,i], rhm_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')


plt.xlabel('$x$')                    
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/sdcard/Download/Matrices/line/rhombus.png')
#plt.show()
