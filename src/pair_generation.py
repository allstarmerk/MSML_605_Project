import json 
from pathlib import path
import numpy as np


def generate_pairs(labels, num_pairs, positive_ratio, seed):
    rng = np.random.default_rng(seed)

    #build identity to image indinces
    id_to_indices = {}
    for idx, label in enumerate(labels):
        id_to_indices.setdefault(int(label), []).append(idx)


    # only for 2+ images can form a pos pair
    multi = {k: np.array(v) for k, v in id_to_indices.items() if len(v) >= 2}
    all_ids = list(id_to_indices.keys())

    n_pos = int(num_pairs * positive_ratio)
    n_neg = num_pairs - n_pos

    pairs, pair_labels = [], []
    
    #pos pairs same identity 2 images
    pos_ids= list(multi.keys())
    for _ in range(n_pos):   #not using index hence naming i _
        identity = pos_ids[rng.integers(len(pos_ids))]
        a, b = rng.choice(multi[identity], size=2, replace=False)
        pairs.append([int(a), int(b)])
        pair_labels.append(1)

    #neg pairs 2 diffrent identity 2 images
    for _ in  range(n_neg):
        i, j = rng.choice(len(all_ids), size = 2, replace=False)
