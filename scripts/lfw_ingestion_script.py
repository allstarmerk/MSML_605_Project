import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('lfw', split='train')
assert isinstance(ds, tf.data.Dataset)

ds = ds.batch(32).prefetch(1)

tfds.benchmark(ds, batch_size=32)
tfds.benchmark(ds, batch_size=32)  # Second epoch much faster due to auto-caching
print(ds)