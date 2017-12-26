import tensorflow as tf
import sys
import math
import os.path
from typing import Tuple, Dict

def int64_feature(values):
  """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def triplet_to_tfexample(triplet):
  return tf.train.Example(features=tf.train.Features(feature={
      'h': int64_feature(triplet[0]),
      'r': int64_feature(triplet[1]),
      't': int64_feature(triplet[2]),
  }))

def write_mapping_file(items_to_ids: Dict[str, int], filename: str, dataset_dir: str):
  """Writes a file with the id mapping.
  Args:
    items_to_ids: A map of (integer) ids to items.
    filename: The filename where the mapping labels are written.
    dataset_dir: The directory in which the labels file should be written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as label_file:
    for item, id_ in items_to_ids.items():
      label_file.write('%d:%s\n' % (id_, item))

def read_mapping_file(dataset_dir, filename):
  """Reads the mapping file and returns a mapping from id to item.
  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    A map from an id to item.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'r') as file:
    lines = file.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  mapping = {}
  for line in lines:
    index = line.index(':')
    mapping[int(line[:index])] = line[index+1:]
  return mapping

def dataset_exists(dataset_dir, num_shards, output_filename) -> bool:
  """validates whether tfrecords for this dataset exists."""
  for split_name in ['train']: # TODO: validate 'validation'
    for shard_id in range(num_shards):
      tfrecord_filename = _gen_dataset_filename(
          dataset_dir, split_name, (shard_id, num_shards), output_filename)
      if not tf.gfile.Exists(tfrecord_filename):
        return False
  return True

def _gen_dataset_filename(dataset_dir: str,
                          split_name: str,
                          shard: Tuple[int, int],
                          tfrecord_filename: str) -> str:
  """returns a path string based on shard and args."""
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard[0], shard[1])
  return os.path.join(dataset_dir, output_filename)

def convert_dataset(split_name: str,
                    loc: int,
                    reader,
                    entities_names_to_ids: Dict[str, int],
                    relations_to_ids: Dict[str, int],
                    dataset_dir: str,
                    tfrecord_filename: str,
                    num_shards: int = 1):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filename: The absolute path to a triplet file (TSV).
    entities_names_to_ids: A dictionary from entity names (strings) to ids (integers).
    relations_to_ids: A dictionary from entity names (strings) to ids (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(loc / float(num_shards)))

  for shard_id in range(num_shards):
    output_filename = _gen_dataset_filename(
        dataset_dir, split_name, (shard_id, num_shards), tfrecord_filename=tfrecord_filename)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_ndx = shard_id * num_per_shard
      end_ndx = min((shard_id+1) * num_per_shard, loc)
      for i in range(start_ndx, end_ndx):
        sys.stdout.write('\r>> Writing triplet %d/%d shard %d' % (
            i+1, loc, shard_id))
        sys.stdout.flush()

        head, relation, tail = next(reader)
        triplet = (entities_names_to_ids[head],
                   relations_to_ids[relation],
                   entities_names_to_ids[tail])

        example = triplet_to_tfexample(triplet)
        tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()
