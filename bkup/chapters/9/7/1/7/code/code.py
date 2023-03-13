import numpy as np
import matplotlib.pyplot as plt
import math

def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB

#To construct triangle APD and BPE
c=8
b=8.2
theta=np.pi*7/36
A=np.array([0,0])
e1=np.array([1,0])
B=c*e1
P=(A+B)/2
Q=b*np.array([np.cos(np.pi-theta),np.sin(np.pi-theta)])
D=b*np.array(([np.cos(theta),np.sin(theta)]))
E=B+Q

m_1 = D-P
m_2 = B-P
n_1 = E-P
n_2 = A-P

m1 = np.linalg.norm(m_1)
m2 = np.linalg.norm(m_2)
n1 = np.linalg.norm(n_1)
n2 = np.linalg.norm(n_2)

#Finding angles
angDPB=np.degrees(np.arccos(np.dot(m_1,m_2)/(m1*m2)))
angEPA=np.degrees(np.arccos(np.dot(n_1,n_2)/(n1*n2)))
if (round(angDPB) == round(angEPA)):
  print("∠ EPA = ∠ DPB")
   
#Generating lines
x_AB = line_gen(A,B)
x_PD = line_gen(P,D)
x_AD = line_gen(A,D)
x_PE = line_gen(P,E)
x_BE = line_gen(B,E)
x_AQ = line_gen(A,Q)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_PD[0,:],x_PD[1,:],label='$PD$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_PE[0,:],x_PE[1,:],label='$PE$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_AQ[0,:],x_AQ[1,:],label='$AQ$',linestyle="--")

#Labeling the coordinates
tri_coords = np.vstack((A,B,P,D,E,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','P','D','E','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$xaxis$')
plt.ylabel('$yaxis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/9.7.1.7/figs/fig.png')
plt.show()

