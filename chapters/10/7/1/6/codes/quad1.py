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


A = []
for i in range(2):
    A.append(float(input("Enter the x and y coordinate for A:")))
A = np.array(A)  

B = []
for i in range(2):
    B.append(float(input("Enter the x and y coordinate for B:")))
B = np.array(B)

C = []
for i in range(2):
    C.append(float(input("Enter the x and y coordinate for C:")))
C = np.array(C)

D = []
for i in range(2):
    D.append(float(input("Enter the x and y coordinate for D:")))
D = np.array(D)

#Enter the points here for manual entring
# A = np.array(([4,5]))
# B = np.array(([7,6]))
# C = np.array(([4,3]))
# D = np.array(([1,2]))

M1 = np.array([(B-A),(C-D)])
M2 = np.array([(C-B),(D-A)])
M3 = np.array([(B-A),(C-B)])
M4 = np.array([(C-B),(D-C)])

R1 = np.linalg.matrix_rank(M1)
R2 = np.linalg.matrix_rank(M2)
R3 = np.linalg.matrix_rank(M3)
R4 = np.linalg.matrix_rank(M4)


#orthogonality of sides
orth1 = np.matmul((np.transpose(B-A)),(C-B))
orth2 = np.matmul((np.transpose(C-A)),(D-B))


#Type of Quadilateral
if R1 == 1 and R2 == 1:
    if orth1 == 0:
        if orth2 == 0:
            print("Square")
        else:
            print("Rectangle")
    elif orth1 != 0 and orth2 == 0:
        print("Rhombus")
    else:
        print("parallelogram")
elif R1==0 or R2==0:
    print("Trapezium")
elif R3==1 or R2 == 1:
    print("Quadilateral cannot be formed")
else:      
    print("irregular quadilateral")      
 

#line generation
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)


plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')


sqr_vert = np.vstack((A,B,C,D)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['A(4,5)','B(7,6)','C(4,3)','D(1,2)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')


plt.xlabel('$x$')                    
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/sdcard/Download/latexfiles/quad3.png')
plt.show()
