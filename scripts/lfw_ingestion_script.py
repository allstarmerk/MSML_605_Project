import sys
import os


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.data_ingestion import (
    load_config,
    load_lfw,
    save_mainifest
)
from src.pair_generation import ( 
    generate_all_splits
)


""""
ds = tfds.load('lfw', split='train')
assert isinstance(ds, tf.data.Dataset)
print(ds)
"""
# 1.load config
data_config = load_config()
#2. load LFW and split

data_splits, data_manifest = load_lfw(data_config)
#3.save manifest
save_mainifest(data_manifest, data_config["paths"]["manifest_path"])
# 4. generate pairs
splits_labels = {name: data[1] for name, data in data_splits.items()}
generate_all_splits(data_config, splits_labels)
#5. then summary to printout what was produced  
print("\n── Summary ───────────────────────────────")
print(f"  Total images : {data_manifest['total_examples']}")
print(f"  Identities   : {data_manifest['num_identites']}")
for split, count in data_manifest["split_counts"].items():
    print(f"  {split:<6} images: {count}")
print("\nDone.")