from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import sys
import IPython.display as display

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_func(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
  return tf.reshape(tf_string, ()) # The result is a scalar


# Create a description of the features.
feature_description = {
    'feature0': tf.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
}
def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  # return tf.parse_single_example(example_proto, feature_description)
  return tf.parse_example(example_proto, feature_description)

# the number of observations in the dataset
n_observations = int(1e5)

# boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)

# integer feature, random from 0 .. 4
feature1 = np.random.randint(0, 5, n_observations)

# string feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)

# This is an example observation from the dataset.

# example_observation = []
#
# output_file = './test.tfrecord'
# writer = tf.python_io.TFRecordWriter(output_file)
# for i in range(n_observations):
#     serialized_example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
#     print(serialized_example)
#
#     # example_proto = tf.train.Example.FromString(serialized_example)
#     # print(example_proto)
#     writer.write(serialized_example)
# writer.close()
# sys.stdout.flush()

features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print(features_dataset)

# for f0,f1,f2,f3 in features_dataset.take(1):
#   print(f0)
#   print(f1)
#   print(f2)
#   print(f3)
serialized_features_dataset = features_dataset.map(tf_serialize_example)
print(serialized_features_dataset)

filename = 'test-1.tfrecord'
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(serialized_features_dataset)
# sys.stdout.flush()

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))

parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))