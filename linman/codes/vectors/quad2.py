import numpy as np
import matplotlib.pyplot as plt
import math
#from coeffs import *
import subprocess
import shlex

#sides of triangle
a = 3
b = 5
c = 6

#coordinates of A
x=(a**2 + c**2 -b**2)/(2*a)
y=np.sqrt(c**2-x**2)

#generating the line points
def line_gen(A,B):
	len = 20
	dim = A.shape[0]
	x_AB = np.zeros((dim,len))
	lam_1 = np.linspace(0,1,len)
	for i in range(len):
		temp1 = A + lam_1[i]*(B-A)
		x_AB[:,i]=temp1.T
	return x_AB

#vertices
P = np.array([-3,5])
Q= np.array([3,1])
R = np.array([0,3])
S = np.array([-1,-4])



#Generating the lines
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_PR = line_gen(P,R)
x_SQ = line_gen(S,Q)
x_SP = line_gen(S,P)

#plotting the all lines
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$')
plt.plot(x_RS[0,:],x_RS[1,:],label='$RS$')
plt.plot(x_PR[0,:],x_PR[1,:],label='$PR$')
plt.plot(x_SQ[0,:],x_SQ[1,:],label='$SQ$')
plt.plot(x_SP[0,:],x_SP[1,:],label='$SP$')



plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1-0.1), P[1]*(1), 'P')
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0]*(1+0.1), Q[1]*(1), 'Q')
plt.plot(R[0], R[1], 'o')
plt.text(R[0]*(1), R[1]*(1+0.2),'R')
plt.plot(S[0], S[1], 'o')
plt.text(S[0]*(1+0.3), S[1]*(1),'S')



plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('../figures/line/quads/quad2.eps')

plt.show()
