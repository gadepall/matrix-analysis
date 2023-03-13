
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shlex


theta=np.pi/4
r=2*np.sqrt(2)
print("Radius(r)=","{:.2f}".format(r))
print("-------------------------------------")
A=r*(np.array((np.cos(theta),np.sin(theta))))
print("A=",A)
B=np.array(([0, 0]))
print("B=",B)
C=2*r*(np.array((np.cos(theta),0)))
print("C=",C)
D=r*(np.array((np.cos(theta),0)))
print("D=",D)
E=r*(np.array((np.cos(theta),0.5*np.sin(theta))))
print("E=",E)


print("-------------------------------------")
print("Calculation of area of triangle ABC")
AreaABC=0.5*((np.linalg.norm(np.cross((A-B),(C-B)))))
print("ar(ABC) = ","{:.0f}".format(AreaABC))
print("-------------------------------------")
print("Calculation of area of triangle BED")
AreaBED=0.5*((np.linalg.norm(np.cross((E-B),(D-B)))))
print("ar(BED) = ","{:.0f}".format(AreaBED))
print("-------------------------------------")
print("Therefore")
print("Area of triangle BED = 1/4 ( Area of triangle ABC)")
print("HENCE PROVED")

# initialize x and y coordinates
Ex = rd.uniform(2,3)
Ey = 12 - 4*Ex

#x1 = [2, 6, 0, 2, 3, 0, Ex, 6]
#y1 = [4, 0, 0, 4, 0, 0, Ey, 0]
x1=[2,0,4,2,2,2,4,0,2]
y1=[2,0,0,2,0,1,0,0,1]
pts1=['A','B','C','A','D','E','C','B','E']
plt.scatter(x1,y1)
for i, txt in enumerate(pts1):
    plt.annotate(txt, # this is the text
                 (x1[i], y1[i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.plot(x1,y1, color="red")
plt.xlabel('$X-axis$')
plt.ylabel('$Y-axis$')
plt.grid() # minor
plt.axis('equal')

plt.savefig('/sdcard/Ajay/matrix/line/linefig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Ajay/matrix/line/linefig.pdf"))



