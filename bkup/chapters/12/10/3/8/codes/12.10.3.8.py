from math import cos 
import numpy as np
import math
import sys

#The inputs are a(transpose)b and theta

#given a(transpose)b=1/2
atransposeb=1/2

#given theta value as pi/3
theta=np.pi/3

#the formula for a is a={a(transpose)b\cos(theta)}^1/2
a=np.sqrt(atransposeb/np.cos(theta))

#given a is equal to b
b=a

#the result 
#print a 
print('a=',round(a))

#print b
print('b=',round(b))
