import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.linalg import norm


def my_func():
   global a,b,theta,e1,A,B,C,D
def cos_angle(x,y):
  return (norm((x*y))/norm(x)*norm(y))

a=6
b=np.sqrt(29)
theta=np.arcsin(5/np.sqrt(29))
e1=np.array([1,0])
C=a*e1
B=np.array([0,0])
A=b*np.array([np.cos(theta),np.sin(theta)])
D=C+A
AB=B-A
CB=B-C
CD=D-C
DA=A-D
CA=A-C

arm_of_triangle_ABC=[AB,CB,CA]
arm_of_triangle_CDA=[CA,DA,CD]
def common(x,y):
  p=np.any(value for value in x if value in y)
  return p
q=bool(common(arm_of_triangle_ABC,arm_of_triangle_CDA))
print(q)
angle_BAC=cos_angle(AB,CA)
print(angle_BAC)
angle_ACD=cos_angle(CD,CA)
print(angle_ACD)
angle_ACB=cos_angle(CB,CA)
print(angle_ACB)
angle_CAD=cos_angle(DA,CA)
print(angle_CAD)
if(q==True and angle_BAC==angle_ACD and angle_ACB==angle_CAD):
  print("The triangles ABC and CDA are congruent to each other")
else:
  print("The triangles ABC and CDA are not congruent to each other")

fig,ax=plt.subplots(figsize=(5,4))
plt.annotate('$p$',xy=(-0.4,-1),xytext=(3,8),arrowprops=dict(arrowstyle='<->',color='b'))
plt.annotate('$m$',xy=(-1,-0.1),xytext=(10,-0.1),arrowprops=dict(arrowstyle='<->',color='b'))
plt.annotate('$l$',xy=(-1,5),xytext=(10,5),arrowprops=dict(arrowstyle='<->',color='b'))
plt.annotate('$q$',xy=(5.62,-1),xytext=(9,8),arrowprops=dict(arrowstyle='<->',color='b'))
plt.plot([2.02,6],[5,0],color='b')
plt.text(1.5,5.5,r'$A$')
plt.text(-0.5,0,u'$B$')
plt.text(6.5,0,r'$C$')
plt.text(8.5,5.5,r'$D$')
plt.xlim(-2,12)
plt.ylim(-2,10)
plt.grid()
plt.show()

