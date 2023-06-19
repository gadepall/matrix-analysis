import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#line generation function
def line_gen(P,Q):
    len1=10
    dim = P.shape[0]
    print(dim)
    x_PQ = np.zeros((dim,len1))
    print(x_PQ)
    lam_1 = np.linspace(0,1,len1)
    print(lam_1)
    for i in range(len1):
        temp1 = P + lam_1[i]*(Q-P)
        x_PQ[:,i] = temp1.T
    return x_PQ
P = np.array(([2,1]))
Q = np.array(([1,-3]))
R= (Q-2*P)/(-1)
print(R)
#line generation 
x_PQ = line_gen(P,Q)
x_PR = line_gen(P,R)
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$ PQ line $')
plt.plot(x_PR[0,:],x_PR[1,:],label='$ PR line $')
sqr_vert = np.block([[P],[R],[Q]]).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['P(2,1)','R(3,5)','Q(1,-3)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
plt.xlabel('$x$')                    
plt.ylabel('$y$')
plt.grid()
plt.axis('equal')
plt.savefig('/storage/self/primary/Download/codes/math/9/figs/line.png')   

