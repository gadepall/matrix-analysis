import numpy as np
import math
import mpmath as mp
from numpy import linalg as LA
import subprocess
import shlex

#Points a and b such that |a| = 8|b| and (a-b).(a+b) = 8
b = np.array(([2/(3*np.sqrt(7)),2/(3*np.sqrt(7))]))
a = np.array(([16/(3*np.sqrt(7)),16/(3*np.sqrt(7))]))

dotProd = np.dot(a-b, a+b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

print("(a-b).(a+b) is ", dotProd)
print("Norm of a is ", norm_a)
print("Norm of b is ", norm_b)
# |a| is k times of |b|, calculating k:
print("k is ", norm_a/norm_b)
