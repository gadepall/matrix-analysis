import numpy as np
import math as ma

# Define the vector
P = np.array(np.round_([2,1,1]))
A  = np.array(np.round_([3,5,-1]))
B  = np.array(np.round_([4,3,-1]))
m=A-B


if (m.T)@P==0:
    print('Hence,it satisfies both the above conditions, shows that line passing through origin is perpendiclar to the line passing through points A and B')
else:
    print('Not perpendicular')
