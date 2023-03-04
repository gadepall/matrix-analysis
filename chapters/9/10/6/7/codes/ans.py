import numpy as np
import matplotlib.pyplot as plt

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

A = np.array([1,0])
B = np.array([0,1])
C = np.array([-1,0])
D = np.array([0,-1])
O = np.array([0,0])

print("AC and BD are are diameters")

if((A+C).all() == (B+D).all()):
    print("AC and BD bisect each other")

#parallel condition
if((B - A).all() == (C - D).all()):
  #rectangle condition
  if((A-B)@(B-C)==0):
    print("ABCD is a rectangle")

#plot a circle with AC and BD as diameters
x_circ = circ_gen(O,np.linalg.norm((A-C)/2))
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

x_AB = line_gen(A,B)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

x_BC = line_gen(B,C)
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')

x_CD = line_gen(C,D)
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')

x_DA = line_gen(D,A)
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')

x_Ac = line_gen(A,C)
plt.plot(x_Ac[0,:],x_Ac[1,:],label='$AC$')

x_BD = line_gen(B,D)
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')

#name the points A,B,C,D,O in plot
plt.text(A[0]*(1+0.1), A[1]*(1-0.1) , 'A')
plt.text(B[0]*(1+0.1), B[1]*(1-0.1) , 'B')
plt.text(C[0]*(1+0.1), C[1]*(1-0.1) , 'C')
plt.text(D[0]*(1+0.1), D[1]*(1-0.1) , 'D')
plt.text(O[0]*(1+0.1), O[1]*(1-0.1) , 'O')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/9.10.6.7/figs/circle.png')
