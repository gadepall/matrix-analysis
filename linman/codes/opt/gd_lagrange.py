#Code by Amey Waghmare, 
#Jan 17, 2020
#Released under GNU GPL
#Gradient Descent with vectors
import numpy as np

iters = 1000
alpha = 0.01
x_old = np.array([[0],[0]])

P = np.array([[3],[-5]])
precision = 0.00000001
n = np.array([[3],[-4]])
c = 26

i = 0
while(i<=iters):
    x_new = x_old - alpha*2*(x_old - P)
    x_old = x_new
    err = n.T@x_new -c
    #print(err)
    if err > precision:
        break
    #print(n.T@x_new)

    cost = (x_new - P).T@(x_new - P)
    i=i+1


print(x_new)
print("Distance is",cost**(1/2))
#print("Number of iterations",i)
