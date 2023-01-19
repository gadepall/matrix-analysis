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
A = np.array(([-2,2]))
B = np.array(([2,8]))
n1=3/1
n2=2/2
n3=1/3
R1= (B+n1*A)/(1+n1) # calculating the coordinate points of R which divides the join between the two points
R2= (B+n2*A)/(1+n2)
R3= (B+n3*A)/(1+n3)

print(R1)

print(R2)

print(R3)


#line generation 
x_AR1=line_gen(A,R1)
x_R1R2=line_gen(R1,R2)
x_R2R3=line_gen(R2,R3)
x_R3B =line_gen(R3,B)

plt.plot(x_AR1[0,:],x_AR1[1,:],label=' A$R_1$ line ')
plt.plot(x_R1R2[0,:],x_R1R2[1,:],label=' $R_1$$R_2$  line ')
plt.plot(x_R2R3[0,:],x_R2R3[1,:],label=' $R_2$$R_3$ line ')
plt.plot(x_R3B[0,:],x_R3B[1,:],label=' $R_3$B  line ')

sqr_vert = np.vstack((A,R1,R2,R3,B)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['A(-2,2)','$R_1(-1,7/2)$','$R_2(0,5)$','$R_3(1,13/2)$','B(2,8)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
plt.xlabel ('$x$')                    
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
#if using termux
#plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/10.7.2.9/figs/10.7.2.9.png')
#subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/10.7.2.9/figs/10.7.2.9.png"))
plt.show()
