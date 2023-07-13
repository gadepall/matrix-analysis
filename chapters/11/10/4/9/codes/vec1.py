import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.linalg import norm
def my_fun():
  global A,B,C
A=np.array([3,1,-2])
B=np.array([5,2,-3])
C=np.array([2,-1,-3])

result1=np.block([[A],[B],[C]])
result2=np.linalg.det(result1)
print(result1)
print(round(result2))


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
plt.plot(x_PS[0,:],x_PS[1,:],color='g',label='$3x+y-2=0$')
plt.plot(x_MN[0,:],x_MN[1,:],color='r',label='$px+2y-3=0$')
plt.plot(x_QR[0,:],x_QR[1,:],color='b',label='2x-y-3=0')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid()
plt.show()



