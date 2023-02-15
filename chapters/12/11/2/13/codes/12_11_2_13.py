import numpy as np


dir_vec_1 = np.array([7,-5,1])
dir_vec_2 = np.array([1,2,3])

pt_1 = np.array([5,-2,0])
pt_2 = np.array([0,0,0])

print('Inner product between the two direction vectors: ',dir_vec_1.T @ dir_vec_2)

