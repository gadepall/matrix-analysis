import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.linalg import norm
def my_fun():
  global n1,n2,n3,c1,c2,c3
p=5
n1=np.array([3,1])
n2=np.array([p,2])
n3=np.array([2,-1])
c1=2
c2=3
c3=3
result1=np.block([[n1,c1],[n2,c2],[n3,c3]])
print(np.linalg.matrix_rank(result1))


fig,ax=plt.subplots(figsize=(5,2.7))
def line_gen(A,B):
  len=10
  dim=A.shape[0]
  x_AB=np.zeros((dim,len))
  lam_1=np.linspace(0,1,len)
  for i in range(len):
    temp1=A+lam_1[i]*(B-A)
    x_AB[:,i]=temp1.T
  return x_AB
Q=np.array([0,2])
R=np.array([9,-25])
M=np.array([0,1.5])
N=np.array([9,-22.5])
P=np.array([0,-3])
S=np.array([9,15])
x_QR=line_gen(Q,R)
x_MN=line_gen(M,N)
x_PS=line_gen(P,S)
plt.plot(x_PS[0,:],x_PS[1,:],color='g',label='(3  1)x=2')
plt.plot(x_MN[0,:],x_MN[1,:],color='r',label='(p  2)x=3')
plt.plot(x_QR[0,:],x_QR[1,:],color='b',label='(2  -1)x=3')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid()
plt.show()

