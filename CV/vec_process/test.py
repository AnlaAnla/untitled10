import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 3,2,1])

top_k = 3

partitioned_indices = np.argpartition(data, -top_k)
top_k_indices_arr = partitioned_indices[-top_k:][np.argsort(data[partitioned_indices[-top_k:]])][::-1]
