from math import cos 
import numpy as np
import math
import sys

#consider veca and the vecb be the two vectors
#veca=np.array([4,3])
#vecb=np.array([5,12])
veca=[]
vecb=[]
n=int(input("enter the length of the array="))
for i in range(n):
    veca.append(int(input("enter the values for veca=")))
    vecb.append(int(input("enter the values for vecb=")))
veca=np.array(veca)
vecb=np.array(vecb)
print(veca)
print(vecb)


#the range of theta values as from 0 to 270 with step of 90 means 0,90,180
for i in np.arange(0,(2*np.pi),(np.pi/6)):

    #the formula for a(transpose)b=cos(theta)*(norm_a)*(norm_b)
    atransposeb=(np.cos(i))*(np.linalg.norm(veca))*(np.linalg.norm(vecb))

  #condition a(transpose)b>=0
    if atransposeb>=0:

      #print if condition is true 
        print("condition satisfied by angle=",round(np.degrees(i)),"for atransposeb=",atransposeb)
    else:

      #print if condition s false
        print("condition not satisfied by angle=",round(np.degrees(i)),"for atransposeb=",atransposeb)
