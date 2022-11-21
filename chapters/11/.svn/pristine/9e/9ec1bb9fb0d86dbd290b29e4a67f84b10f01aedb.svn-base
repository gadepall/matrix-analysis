import numpy as np
import matplotlib.pyplot as plt
r = 5
coeff = [1,-4,-12]
A = np.roots(coeff)
O = np.array((A[0],0))
O1 = np.array((A[1],0))
def circ_gen(O,r):
    len = 50
    theta = np.linspace(0,2*np.pi,len)
    xc = np.zeros((2,len))
    xc[0,:] = r*np.cos(theta)
    xc[1,:] = r*np.sin(theta)
    xc = (xc.T + O).T
    return xc
xc = circ_gen(O,5)
xc1 = circ_gen(O1,5)
plt.plot(xc[0,:],xc[1,:])
plt.plot(xc1[0,:],xc1[1,:])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/sdcard/Download/Matrices/conic/conic.png')
#plt.show()
