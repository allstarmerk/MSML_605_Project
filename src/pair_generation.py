import json 
from pathlib import Path
import numpy as np

# src/pair_gen generates Deterministic pos/neg  pairs and saved as .npy 
# pos pair is two images and labe =1 becasue they are within similarity score and are "Same"
#neg pai is 2 images and label =0 because two images are diffrent bellow similarity score threshold



#builds verification pairs from a label array "labels" are a (N,) int array of identity labels
# "num pairs" are the total pairs we are generating
# "seed" the random seed defined in config file /config/dataset.yaml
#We return the "pairs" indices into the iamge array and "pair labels" labels defined above line 7
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
        id_a, id_b = all_ids[i], all_ids[j]
        a = id_to_indices[id_a][rng.integers(len(id_to_indices[id_a]))]
        b = id_to_indices[id_b][rng.integers(len(id_to_indices[id_b]))]
        pairs.append([int(a), int(b)])
        pair_labels.append(0)

    pairs = np.array(pairs, dtype=np.int32)
    pair_labels = np.array(pair_labels, dtype=np.int32)

    # shuffle 

    perm = rng.permutation(len(pairs))
    return pairs[perm], pair_labels[perm]

def save_pairs(pairs, pair_labels, split_name, pairs_dir):
    out = Path(pairs_dir)  #saves pair and label as a .npy file
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / f"{split_name}_pairs.npy", pairs)
    np.save(out / f"{split_name}_labels.npy", pair_labels) 

    print(f"[{split_name}] {len(pairs)} pairs saved to {out}")
    return {
        "split": split_name,
        "total_pairs": len(pairs),
        "positive_pairs": int(pair_labels.sum()),
        "negative_pairs": int((pair_labels == 0).sum()),

    }
    #generate and save pairs for every split. Takes in -config(loaded yaml config dictionary) -splits_labled(dictionary of split names to a labled array) 
def generate_all_splits(config, splits_labels): 
    pair_cfg = config["pairs"]
    pairs_dir = config["paths"]["pairs_dir"]
    base_seed = pair_cfg["seed"]
    all_meta = {}

    for split_name, labels in splits_labels.items():
        num_pairs = pair_cfg["num_pairs_per_split"][split_name]
        split_seed = base_seed + hash(split_name) % 1000  # derivin a deterministic seed split like project requirments call for 

        pairs, pair_labels = generate_pairs(
            labels = labels,
            num_pairs = num_pairs,
            positive_ratio = pair_cfg["positive_ratio"],
            seed = split_seed,
        )
        all_meta[split_name] = save_pairs(pairs, pair_labels, split_name, pairs_dir)

    #save manifest for the pairs
    manifest_path = Path(pairs_dir) / "pairs_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2) 
    print(f"Pairs manifest saved to {manifest_path}")
        
    return all_meta