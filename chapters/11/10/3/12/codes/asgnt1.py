import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.linalg import norm
def my_func():
  global m1,m21,m22,P,theta,p,q,r,a,b,c,coeff1,coeff2
m1=2
P=np.array([2,3])
theta=60*(np.pi/180)
a=(np.cos(theta))**2
b=4**2-4*(5*a-4)*(5*a-1)
c=2*(5*a-4)
m21=(4+np.sqrt(b))/c
m22=(4-np.sqrt(b))/c
print(m21)
print(m22)
11*m21**2+16*m21-1==0
coeff1=[11,16,-1]
r=np.roots(coeff1)
print(r)
11*m22**2+16*m22-1==0
coeff2=[11,16,-1]
q=np.roots(coeff2)
print(q)


def line_gen(A,B):
  len=100
  dim=A.shape[0]
  x_AB=np.zeros((dim,len))
  lam_1=np.linspace(0,1,len)
  for i in range(len):
    temp1=A+lam_1[i]*(B-A)
    x_AB[:,i]=temp1.T
  return x_AB

 

fig,ax=plt.subplots(figsize=(5,3))
Q=np.array([0,-0.3])
R=np.array([8.5,15.5])
M=np.array([0,6.3])
N=np.array([8.5,-5.6])
K=np.array([0,3.4])
L=np.array([8.5,3.5])
x_MN=line_gen(M,N)
x_QR=line_gen(Q,R)
x_KL=line_gen(K,L)
plt.plot(x_QR[0,:],x_QR[1,:],color='b',label='(2    -1)x=1')
plt.plot(x_MN[0,:],x_MN[1,:],color='m',label='(-8-5√3)/11     -1)x=(-49-16√3)/11')
plt.plot(x_KL[0,:],x_KL[1,:],color='r',label='(-8+5√3)/11     -1)x=(-49+16√3)/11')
plt.legend()
plt.grid()
plt.show()

