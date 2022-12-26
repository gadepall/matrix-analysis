import numpy as np
import math
import mpmath as mp
from numpy import linalg as LA
import subprocess
import shlex

#Given points
x = np.array(([2,3]))
a = np.array(([1/2,math.sqrt(3)/2]))   # Unit vector

dotProd = np.dot(x-a, x+a)
norm_x = np.linalg.norm(x)

print("(x-a).(x+a) is ", dotProd)
print("Norm of x is ", norm_x)
