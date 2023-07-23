import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.linalg import norm

def my_func():
  global theta,alpha,beta,O,P,Q,R,angle_OPR

r=1
O=np.array([0,0])
theta=100*np.pi/180
theta2=165*np.pi/180
theta3=5*np.pi/180

a=(theta3+theta2)/2
b=(theta2/2)
c1=(np.tan(b)*((np.cos(theta)-np.cos(a))/(np.cos(theta)+np.cos(a))))
c2=(np.tan(b)*((np.cos(theta)+np.cos(a))/(np.cos(theta)-np.cos(a))))
d1=np.arctan(c1)
d2=np.arctan(c2)
theta11=2*d1*180/np.pi
theta12=2*d2*180/np.pi

print(theta11)
print(theta12)


P=r*np.array([np.cos(theta2),np.sin(theta2)])
R=r*np.array([np.cos(theta3),np.sin(theta3)])
Q=r*np.array([np.cos(theta11),np.sin(theta11)])
PO=O-P
PR=R-P
M=np.array([-1,0])
N=np.array([1,0])

angle_OPR=(180-(2*(theta*180/np.pi)))/2
print("The value of angle OPR is",angle_OPR,"degree")


fig,ax = plt.subplots(figsize=(5,5))
def line_gen(A,B):
  len=100
  dim=A.shape[0]
  x_AB=np.zeros((dim,len))
  lam_1=np.linspace(0,1,len)
  for i in range(len):
    temp1=A+lam_1[i]*(B-A)
    x_AB[:,i]=temp1.T
  return x_AB

def circ_gen(O,r):
 len = 50
 theta = np.linspace(0,2*np.pi,len)
 x_circ = np.zeros((2,len))
 x_circ[0,:] = r*np.cos(theta)
 x_circ[1,:] = r*np.sin(theta)
 x_circ = (x_circ.T + O).T
 return x_circ
x_circ= circ_gen(O,r)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_OP = line_gen(O,P)
x_OR = line_gen(O,R)
x_PR = line_gen(P,R)
x_MN = line_gen(M,N)
x_OQ = line_gen(O,Q)





plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_PR[0,:],x_PR[1,:],label='$PR$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$')
plt.plot(x_OR[0,:],x_OR[1,:],label='$OR$')
plt.plot(x_OP[0,:],x_OP[1,:],label='$OP$')
plt.plot(x_MN[0,:],x_MN[1,:],label='$MN$')
plt.plot(x_OQ[0,:],x_OQ[1,:],label='$OQ$')

plt.text(0,0,u'$O$')
plt.text(-1,0.15,u'$P$')
plt.text(0.96,0.15,r'$R$')
plt.text(0.6,-1.05,r'$Q$')
plt.text(-1,0,r'$M$')
plt.text(1,0,r'$N$')

plt.title('Required figure with θ1=175◦')
plt.xlabel('Values of angles')
plt.legend()
plt.grid()
plt.show()

