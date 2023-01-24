import numpy as np

# Creating two vectors
# We have inserted elements of int type
A =np.array(( [1, -1, 2]))
B =np.array(( [3, 4, -2]))
C =np.array(( [0, 3, 2]))
D =np.array(( [3, 5, 6]))

X1=A-B
X2=C-D
z=(X1.T)@X2
print(z)



