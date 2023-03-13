import numpy as np

#Inner product definition
#Returns inner product of
#B-A and C-A
def inner_prod(A,B,C):
    return (B-A).T@(C-A)

#Create column vectors for points
A = np.array([[3.0],[-4.0],[-4.0]])
B = np.array([[2.0],[-1.0],[1.0]])
C = np.array([[1.0],[-3.0],[-5.0]])

#Take inner products one pair at a time
p = inner_prod(A,B,C)
q = inner_prod(B,C,A)
r = inner_prod(C,A,B)

#Check and print output
print("Triangle ABC is", end=" ")
if p == 0:
    print("right angled at A.")
elif q == 0:
    print("right angled at B.")
elif r == 0:
    print("right angled at C.")
else:
    print("not right angled.")
