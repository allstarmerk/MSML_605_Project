import sys
import os


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, root_path)

from src.data_ingestion import (
    load_config,
    load_lfw
)


""""
ds = tfds.load('lfw', split='train')
assert isinstance(ds, tf.data.Dataset)
print(ds)
"""

data_config = load_config()

data_splits, data_manifest = load_lfw(data_config)