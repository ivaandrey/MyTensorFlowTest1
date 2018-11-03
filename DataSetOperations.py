# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
import random
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
from ReadPklFiles import ReadAllPklFile


# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'C:\\Andrey\\DeepLearning\\TensorF\\PointTargetProject\\MydataDir\\'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


@deprecated(None, 'Please use tf.data to implement this functionality.')
def extract_images(all_pckl_files,NormalizeImages):
  """
  Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    all_pckl_files: Data from all pickle files

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """

  # find image size and num of images
  first_image=all_pckl_files[0]
  if 'ImageRow' in first_image:
    rows= first_image['ImageRow']
  if 'ImageCol' in first_image:
    cols = first_image['ImageCol']
  if 'ImageWidth' in first_image:
      rows = first_image['ImageWidth']
  if 'ImageHeight' in first_image:
      cols = first_image['ImageHeight']
  if 'ImageChannels' in first_image:
    channels = first_image['ImageChannels']
  else:
    channels=1

  eps=1e-8
  numOfImages=len(all_pckl_files)
  data=numpy.empty((numOfImages,rows,cols,channels), dtype=float, order='C')
  r = numpy.empty((rows, cols,1), dtype=float, order='C')
  g = numpy.empty((rows, cols, 1), dtype=float, order='C')
  b = numpy.empty((rows, cols, 1), dtype=float, order='C')
  ImageVector= numpy.empty((rows*cols*channels,1), dtype=float, order='C')
  image_reshaped= numpy.empty((rows, cols, channels), dtype=float, order='C')
  image_norm = numpy.empty((rows, cols, channels), dtype=float, order='C')
  for image_ind,image_data in zip(range(numOfImages),all_pckl_files):
    ImageVector=image_data['ImageVector']
    image_reshaped = ImageVector.reshape(rows, cols, channels)

    # Image normalization
    if NormalizeImages and channels==1:
      # Normalize grayscale image
      meanImage=numpy.mean(ImageVector,dtype=float)
      stdImage=numpy.std(ImageVector,dtype=float)
      image_norm=(image_reshaped-meanImage)/stdImage
    elif NormalizeImages and channels == 3:
      # Normalize RGB image
      r = image_reshaped[:, :, 0]
      g = image_reshaped[:, :, 1]
      b = image_reshaped[:, :, 2]
      sum = b + g + r
      image_norm[:, :, 0] = r / (sum+eps) * 255.0
      image_norm[:, :, 1] = g / (sum+eps) * 255.0
      image_norm[:, :, 2] = b / (sum+eps) * 255.0
    else:
      image_norm=image_reshaped

    data[image_ind,:,:,:]=image_norm
  return data

def extract_additional_data(all_pckl_files):
  """
  Extract additional data of the image as a list of dictionaries

  Args:
    all_pckl_files: Data from all pickle files

  Returns:
    data: list of dictionaries

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """

  # find image size and num of images
  numOfImages = len(all_pckl_files)
  ImageAdditionalData = []
  for image_ind, image_data in zip(range(numOfImages), all_pckl_files):
    """
    CurrImageAdditionalData= {'ImageName': image_data['ImageName'],
                     'ObjectDataValid': image_data['ObjectDataValid'],
                     'ObjectXLoc': image_data['ObjectXLoc'],
                     'ObjectYLoc': image_data['ObjectYLoc'],
                     'ObjectSNR': image_data['ObjectSNR']}
    """
    del image_data['ImageVector']# remove image from the dictionary
    ImageAdditionalData.append(image_data)

  return ImageAdditionalData



@deprecated(None, 'Please use tf.one_hot on tensors.')
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


@deprecated(None, 'Please use tf.data to implement this functionality.')
def extract_labels(all_pckl_files, num_classes,LabelName,one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    all_pckl_files: Data from all pickle files
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.
  """

  # find num of images
  numOfImages = len(all_pckl_files)
  labels = numpy.empty((numOfImages, 1), dtype=numpy.uint8)
  for image_ind, image_data in zip(range(numOfImages), all_pckl_files):
    labelData = image_data[LabelName]
    labels[image_ind] = labelData

  if one_hot:
    return dense_to_one_hot(labels, num_classes)
  return labels

def reshuffle_list(original_list, new_indexes):
  """
  Reshuffle  a list of dictionaries

  Args:
    original_list:original_list
    new_indexes

  Returns:
    resuffled list of dictionaries



  """

  ReshuffledList = []
  for curr_index in new_indexes:
      curr_list_data=original_list[curr_index]
      ReshuffledList.append(curr_list_data)

  return ReshuffledList


class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               additional_data,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
      """
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
        """
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns*depth] (assuming depth == 1)
      if reshape:
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]*images.shape[3])
    self._images = images
    self._labels = labels
    self.additional_data=additional_data
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
      self._additional_data =reshuffle_list(self.additional_data, perm0)
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      additional_data_rest_part = self._additional_data[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
        self._additional_data = reshuffle_list(self.additional_data, perm)
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      additional_data_new_part = self._additional_data[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0),additional_data_rest_part.append(additional_data_new_part)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end],self._additional_data[start:end]


@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def read_data_sets(train_dir,test_dir,LabelName,NormalizeImages,
                   num_of_classes=2,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_percent=10,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  all_pckl_train_files=ReadAllPklFile(train_dir)
  all_pckl_test_files = ReadAllPklFile(test_dir)


  random.seed(3)

  train_images = extract_images(all_pckl_train_files,NormalizeImages)
  train_labels = extract_labels(all_pckl_train_files, num_of_classes,LabelName,one_hot=one_hot)
  train_images_additional_data = extract_additional_data(all_pckl_train_files)

  perm = numpy.arange(len(train_images))
  numpy.random.shuffle(perm)
  train_images = train_images[perm]
  train_labels = train_labels[perm]
  train_images_additional_data=reshuffle_list(train_images_additional_data,perm)


  test_images = extract_images(all_pckl_test_files,NormalizeImages)
  test_labels = extract_labels(all_pckl_test_files, num_of_classes,LabelName,one_hot=one_hot)
  test_images_additional_data = extract_additional_data(all_pckl_test_files)

  validation_size=round(validation_percent*len(train_images)/100)
  if not 0 < validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                     .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  validation_images_additional_data = train_images_additional_data[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]
  train_images_additional_data = train_images_additional_data[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels,train_images_additional_data, **options)
  validation = DataSet(validation_images, validation_labels,validation_images_additional_data, **options)
  test = DataSet(test_images, test_labels,test_images_additional_data, **options)

  return base.Datasets(train=train, validation=validation, test=test)


@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)
