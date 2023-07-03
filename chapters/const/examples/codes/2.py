import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
def my_func():
  global a,b,c,theta,A,B,C,e1
a=7
b=(436-91*np.sqrt(6)+91*np.sqrt(2))/(52-7*np.sqrt(6)+7*np.sqrt(2))
c=240/(52-7*np.sqrt(6)+7*np.sqrt(2))
theta=(75*np.pi/180)
b+c==13
e1=np.block([1,0])
A=c*(np.block([np.cos(theta),np.sin(theta)]))
B=np.block([0,0])
C=a*e1
print(A)

fig,ax=plt.subplots(figsize=(5,2.7))
ax.set_title("Required triangle")
#plt.plot([0,1.38,7,0],[0,5.18,0,0])#we can use it for getting single color arms of triangle
plt.plot([0,1.38],[0,5.18],color='b')
plt.plot([1.38,7],[5.18,0],color='r')
plt.plot([7,0],[0,0],color='y')
plt.xlim(0,10)
plt.ylim(-2,8)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.text(1.38,5.18,u'$A$')
plt.text(0,0,r'$B$')
plt.text(7,0,r'$C$')

plt.grid()
plt.show()

