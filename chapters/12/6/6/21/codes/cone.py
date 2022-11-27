import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import *
from numpy import linalg as LA

import sympy as sym
from sympy.solvers.solveset import linsolve
m,x,y = sym.symbols('m x y')

#direction vector 
m1 = np.array(([1,m]))
n = np.array(([-m,1]))
v1 = np.array(([0,0]))
v2 = np.array(([0,1]))
V = np.block([[v1],[v2]])
u = np.array(([-2,0]))

q=np.array(([(2-m)/m**2,2/m]))


peq = q@V@q.T + 2*u@q.T

print(peq)
ans = solve(peq,m)
print(ans)
ans = ans[0]




A = np.array(([-1,1],[0,1]))
B = np.array(([1,2]))
q = np.linalg.solve(A,B)
print(q)
q = np.array(([q]))

O=np.array(([0,0]))

mpl.rcParams['lines.color'] ='k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',['k'])
x=np.linspace(-9,9,400)
y=np.linspace(-9,9,400)
x, y = np.meshgrid(x,y)
def axes():
    plt.axhline(0,alpha=.1)
    plt.axvline(0,alpha=.1)
tri_coords = np.vstack((O,q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
a=1
axes()
plt.contour(x,y,(4*a*x-y**2),[0],colors='k')
plt.grid()

x = np.linspace(-9,9,400)
y = ans*x+1
plt.plot(x, y, '-r',label = 'line(y = mx+1)')
plt.title('Parabola')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
plt.savefig('/sdcard/iithfwc/trunk/cone/im.pdf')
