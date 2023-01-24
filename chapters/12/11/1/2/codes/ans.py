#Find the direction cosines of a line which makes equal angles with the coordinate axes.

import numpy as np

#Define the line with direction cosines cosx, cosx, cosx
pm = np.array([+1, -1])

cosx = 1/np.sqrt(3)
l = np.array([cosx, cosx, cosx])

print("Norm of l is", np.linalg.norm(l))
print("Direction cosines of l are", pm*cosx, pm*cosx, pm*cosx)
