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

# print(_bytes_feature(b'test_string'))
# f1 = _bytes_feature(b'test_string')
# print(_bytes_feature(u'test_bytes'.encode('utf-8')))
# f2 = _bytes_feature(u'test_bytes'.encode('utf-8'))
# print(f1 == f2)  # False
# print(_float_feature(np.exp(1)))
# f3 = _float_feature(np.exp(1))
# print(_int64_feature(True))
# f4 = _int64_feature(True)
# print(_int64_feature(1))
# f5 = _int64_feature(1)

# feature = _float_feature(np.exp(1))
# print(feature.SerializeToString())

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

example_observation = []

# data = np.array([[False, 4, b'goat', 0.9876],
#                  [False, 4, b'goat', 0.9876],
#                  [False, 4, b'goat', 0.9876],
#                  [False, 4, b'goat', 0.9876]])

output_file = './test.tfrecord'
writer = tf.python_io.TFRecordWriter(output_file)
for i in range(n_observations):
    serialized_example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    print(serialized_example)

    # example_proto = tf.train.Example.FromString(serialized_example)
    # print(example_proto)
    writer.write(serialized_example)
writer.close()
sys.stdout.flush()