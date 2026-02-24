# where we will do the vectorized cosine and euclidean similarity functions
# first slow usiongpython loop
#then faster vectorized 


import numpy as np

#using vectorized faster version then python loop for cosine and euclidean
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) #keep dim perventing the (N,) to be considered (1,N) like broadcasting would normaly do. This makes it become (N,1) for broadcasting
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    norm_a = np.where(norm_a == 0, 1e-10, norm_a) #doing so no zero vectors 
    norm_b = np.where(norm_b == 0, 1e-10, norm_b)

    a_unit = a / norm_a #(N, D)
    b_unit = b / norm_b # same as above N,D

    return np.einsum("nd , nd to N", a_unit, b_unit) #(N,)


