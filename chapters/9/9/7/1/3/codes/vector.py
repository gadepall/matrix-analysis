import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#line generation function
def line_gen(A,O):
    len=10
    dim = A.shape[0]
    x_AO = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(O-A)
        x_AO[:,i] = temp1.T
    return x_AO
    
a=3
b=8
A = np.array([a,0])
B = np.array([a,b])
C = np.array([2*a,b])
D = np.array([0,0])

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$distance(AB)$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$distance(BC)$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$distance(CD)$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$distance(DA)$')

x = [2.8, 2.8, 3]
y = [0, 0.2, 0.2]
plt.plot(x, y, color='black')

x = [3, 3.2, 3.2]
y = [7.8, 7.8, 8]
plt.plot(x,y, color='black')


sqr_vert = np.vstack((A,B,C,D)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['A','B','C','D']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')

line1 = [[1.5, 0.1], [1.5, -0.1]]  # First line from X to Y
line2 = [[4.5, 7.9], [4.5, 8.1]]  # Second line from Y to Z
plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'r-')  # First line
plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'g-')  # Second line

plt.xlabel('$x$')                    
plt.ylabel('$y$')
#plt.legend(['d1($AB$)='+str(d1),'d2($CD$)='+str(d2),'d3($EF$)='+str(d3)])
plt.grid()
plt.axis('equal')
#if using termux
#plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.1/figs/graph.png')
#subprocess.run(shlex.split("termux-open /sdcard/download/fwcassgn/trunk/fwcassgn/trunk/vectors/7.1/figs/graph.png"))
plt.show()

