import tensorflow as tf
import tensorflow_datasets as tfds

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