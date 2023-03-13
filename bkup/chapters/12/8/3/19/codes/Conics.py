import math 
import matplotlib.pyplot as plt
import sympy as smp
from sympy import *
p= smp.symbols('p')
import numpy as np
x = np.arange(0,3*np.pi-1,0.3)   # start,stop,ste
y = np.sin(x)
z = np.cos(x)
print('The Area bounded by Y-axis is')
print(smp.integrate((smp.cos(p) - smp.sin(p)) ,(p,0,(smp.pi)/4)))    #x = np.linspace(0, 2, 1000)
plt.plot(x,y)
plt.plot(x,z)
plt.xlabel('x values from 0 to 2pi')  # string must be enclosed with quotes '  '
plt.ylabel('sin(x) and cos(x)')
plt.title('Plot of sin and cos from 0 to 2pi')
plt.legend(['sin(x)', 'cos(x)'])      # legend entries as seperate strings in a list
plt.grid(True, which='both')
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.fill_between(x,y,z,where=[(x > -0.5) and (x < 0.9) for x in x])
#if using termux
#plt.savefig('/sdcard/conics.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/fwc/conics.pdf"))
plt.show()
