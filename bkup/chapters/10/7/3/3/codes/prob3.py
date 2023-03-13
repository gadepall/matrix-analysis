import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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





A = np.array(([0,-1]))
B = np.array(([2,1]))
C = np.array(([0,3]))



P=0.5*(A+B)
Q=0.5*(B+C)
R=0.5*(A+C)
ar1=0.5*(np.linalg.norm((P-Q)@(Q-R)))
ar2=0.5*(np.linalg.norm((A-B)@(A-C)))
print(str(int(ar1))+':'+str(int(ar2)))



#line generation with the calculated distances(d1,d2,d3)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RP = line_gen(R,P)




plt.plot(x_AB[0,:],x_AB[1,:],label='$distance(AB)$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$distance(BC)$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$distance(CA)$')
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$distance(CA)$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$distance(CA)$')
plt.plot(x_RP[0,:],x_RP[1,:],label='$distance(CA)$')


sqr_vert = np.vstack((A,B,C,P,Q,R)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['A(0,-1)','B(2,1)','C(0,3)','P(1,0)','Q(1,2)','R(0,1)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')


plt.xlabel('$x$')                    
plt.ylabel('$y$')
#plt.legend(['d1($AB$)='+str(d1),'d2($CD$)='+str(d2),'d3($EF$)='+str(d3)])
plt.grid()
plt.axis('equal')
#if using termux
plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.2.3/figs/trigraph.png')
subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.2.3/figs/trigraph.png"))
plt.show()
