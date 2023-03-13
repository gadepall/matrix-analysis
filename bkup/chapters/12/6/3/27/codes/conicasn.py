import random as rd
import numpy as np
import matplotlib.pyplot as plt
import math

#if using termux
#import subprocess
#import shlex
#end if

#input parameters
slope = 2
m = np.array([[1],[slope]])
n = np.array([[-slope],[1]])
v = np.array([[0,0.5],[0.5,0]])
u = np.array([[0],[-3/2]])
f = -1

vin = np.linalg.inv(v)

f0 = f + u.T @ vin @ u

k = math.sqrt(f0 / (n.T @ vin @ n))

q = vin @ ( k * n - u ) 


mu = ( 1 / ( m.T @ v @ m ) ) @ ( - m.T @ ((v @ q) + u ) ) 

xi = q + mu * m

x =xi[0][0]
y=xi[1][0]
pt=['P']

plt.scatter(x,y)

for i, txt in enumerate(pt):
    plt.annotate(txt, # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

print("The Required Point Of Contact is: ")
print(xi)

x1 = np.linspace(-6,10,100)
y1 = 1 / (x1 - 3) 
plt.plot(x1,y1, color="Blue")

x2 = np.linspace(-6,10,100)
y2 = [slope * (t - x) + y for t in x2]

plt.plot(x2,y2, color="Red")

plt.xlabel('$X-axis$')
plt.ylabel('$Y-axis$')
plt.grid() # minor
plt.show()


plt.savefig('/sdcard/Download/conicfig.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/conicfig.pdf'"))
