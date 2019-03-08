from datetime import datetime
import os
import random
import sys
import threading
import math

import numpy as np
import tensorflow as tf


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer,
                        height, width):
    """Build an Example proto for an example.

    Args:
      orig_filename: string, path to an image file, e.g., '/path/to/example.JPG'
      label_filename: string, path to an image file, e.g., '/path/to/example.JPG'
      orig_image_buffer: string, JPEG encoding of RGB image
      label_image_buffer: string, JPEG encoding of RGB image
      orig_height: integer, image height in pixels
      orig_width: integer, image width in pixels
      label_height: integer, image height in pixels
      label_width: integer, image width in pixels
    Returns:
      Example proto
    """
    channels = 1

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channels': _int64_feature(channels),
        'image': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=1)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='grayscale', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 1

    return image_data, height, width


def _process_image_files_batch(coder, name, img_paths, output_directory, shards_size):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      name: string, unique identifier specifying the data set
      orig_filenames: list of strings; each string is a path to an image file
      label_filenames: list of strings; each string is a path to an image file
      outpul_directory : Directory for output files
      shards_size: integer size of shards for this data set.
    """
    if (shards_size != -1):
        total_shards = int(math.ceil(len(img_paths) / shards_size))
    else:
        total_shards = 1
        shards_size = len(img_paths)
    print("Total Partitions %d, partition size %d " % (total_shards, shards_size))
    for shard in range(total_shards):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, total_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        # shard_counter = 0
        files_in_shard = np.arange(shard * shards_size, min((shard + 1) * shards_size, len(img_paths)), dtype=int)
        for i in files_in_shard:
            path = img_paths[i]

            image_buffer, height, width = _process_image(path, coder)

            example = _convert_to_example(path, image_buffer,
                                          height, width)
            writer.write(example.SerializeToString())
        print("Processed files %d of %d" % (shard * shards_size, len(img_paths)))
        writer.close()
        sys.stdout.flush()
        # shard_counter = 0
    sys.stdout.flush()


def main(image_folder, output_directory, shards_size=-1):
    img_paths = [os.path.join(image_folder, im) for im in os.listdir(image_folder) if
                      os.path.isfile(os.path.join(image_folder, im))]

    coder = ImageCoder()
    print(len(img_paths))
    _process_image_files_batch(coder, "data", img_paths, output_directory, shards_size)


if __name__ == '__main__':
    main(image_folder='./data/images2',
         output_directory='./data/image2-tfrecord',
         shards_size=-1)