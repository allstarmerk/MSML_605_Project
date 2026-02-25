import sys
import os


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from src.data_ingestion import (
    load_config,
    load_lfw,
    save_mainifest
)
from src.pair_generation import ( generate_all_splits)


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

# 4. generate pairs

#5. then summary to printout what was produced  

#need to finish and test