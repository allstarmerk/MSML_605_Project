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


def euclidean_distance(a, b):
    diff = a - b #(N,D)
    return np.sqrt(np.einsum("nd, nd to N", diff, diff)) #(N,)



#doing slow version using loop cosine
def cosine_similarity_loop(a, b):
    n = a.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        dot = float(np.dot(a[i], b[i]))
        norm = float(np.linalg.norm(a[i])) * float(np.linalg.norm(b[i]))
        out[i] = dot / (norm + 1e-10)
    return out

#slow loop for euuclindean

def euclidean_distance_loop(a, b):
    n = a.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = float(np.sqrt(np.sum(a[i] - b[i] ** 2)))
    return out


def flatten_embedding(images):  #flattens pizels and normalizes. image (N, H, W, C) returns (N, H*W*C)
    n = images.shape[0]
    flat = images.reshape(n, -1).astype(np.float32)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-10, norm)
    return flat / norm


