import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('lfw', split='train')
assert isinstance(ds, tf.data.Dataset)
print(ds)