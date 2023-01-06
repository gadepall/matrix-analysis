import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#line generation function
def line_gen(P,Q):
    len=10
    dim = P.shape[0]
    x_PQ = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = P + lam_1[i]*(Q-P)
        x_PQ[:,i] = temp1.T
    return x_PQ





P = np.array(([-1,7]))
Q = np.array(([4,-3]))
n=3/2
R= (Q+n*P)/(1+n) # calculating the coordinate points of R which divides the join between the two points


print(R)



#line generation 
x_PR = line_gen(P,R)
x_RQ = line_gen(R,Q)





plt.plot(x_PR[0,:],x_PR[1,:],label='$ PR line $')
plt.plot(x_RQ[0,:],x_RQ[1,:],label='$ RQ  line $')



sqr_vert = np.vstack((P,R,Q)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['P(-1,7)','R(1,3)','Q(4,-3)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')


plt.xlabel('$x$')                    
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.2/figs/linefig.png')
subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.2/figs/linefig.png"))
plt.show()
