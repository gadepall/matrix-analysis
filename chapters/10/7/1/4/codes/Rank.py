import numpy as np

#Rank of the Matrix
my_matrix = np.array([[5,-2], [6,4], [7,-2]])
print("Matrix")
for row in my_matrix:
  print(row)
rank = np.linalg.matrix_rank(my_matrix)
print("Rank of the Matrix is : ",rank)
