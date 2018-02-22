import tensorflow as tf
import sys
import math
import os.path
from typing import Tuple, Dict
import gentrain
from collections import defaultdict

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

def convert_dataset(num_triplets: int,
                    batch_size: int,
                    reader,
                    entities_to_ids: Dict[str, int],
                    relations_to_ids: Dict[str, int],
                    dataset_dir: str,
                    negative_relation_rate: int,
                    negative_entity_rate: int,
                    bern: bool):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filename: The absolute path to a triplet file (TSV).
    entities_to_ids: A dictionary from entity names (strings) to ids (integers).
    relations_to_ids: A dictionary from entity names (strings) to ids (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """

  gentrain.init_buff(num_triplets, len(entities_to_ids), len(relations_to_ids))

  for head, relation, tail in reader:
    triplet = (entities_to_ids[head],
                relations_to_ids[relation],
                entities_to_ids[tail])
    read_triplet = gentrain.feed(triplet)
    sys.stdout.write('\r>> Read triplet %d' % (read_triplet))
    sys.stdout.flush()

  gentrain.freq()

  sys.stdout.write('\r>> Proceeding to generate positive triplets')
  sys.stdout.flush()
  with tf.python_io.TFRecordWriter("positive_triplets.tfrecord") as writer:
    for shard in range(read_triplet // batch_size + 1):
      triplets = []
      for i in range(batch_size):
        sys.stdout.write('\r>> Writing triplet %d/%d shard %d' % (
            i+1, batch_size, shard))

        triplets.append(gentrain.yield_triplet())

      ex = triplets_to_tf_example(triplets)
      writer.write(ex.SerializeToString())

  sys.stdout.write('\r>> Proceeding to generate negative triplets')
  sys.stdout.flush()
  with tf.python_io.TFRecordWriter("negative_triplets.tfrecord") as writer:
    for shard in range(read_triplet // batch_size + 1):
      triplets = []
      for i in range(batch_size):
        sys.stdout.write('\r>> Writing triplet %d/%d shard %d' % (
            i+1, batch_size, shard))

        for t in gentrain.yield_neg_triplets(negative_entity_rate, negative_relation_rate, bern):
          triplets.append(t)

      ex = triplets_to_tf_example(triplets)
      writer.write(ex.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def triplets_to_tf_example(triplets):
  ex = tf.train.SequenceExample()
  # context for sequence example
  triplets_length = len(triplets)
  ex.context.feature["length"].int64_list.value.append(triplets_length)

  # Feature lists for sequential features of our example
  head_tokens = ex.feature_lists.feature_list["heads"]
  relation_tokens = ex.feature_lists.feature_list["relations"]
  tail_tokens = ex.feature_lists.feature_list["tails"]

  for triplet in triplets:
    h, r, t, _ = triplet
    head_tokens.feature.add().int64_list.value.append(h)
    relation_tokens.feature.add().int64_list.value.append(r)
    tail_tokens.feature.add().int64_list.value.append(t)

  return ex

def parse_triplets_from_sequence_example(ex):
  '''
  Explain to TF how to go froma  serialized example back to tensors
  :param ex:
  :return: A dictionary of tensors, in this case {seq: The sequence, length: The length of the sequence}
  '''
  context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    "heads": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "relations": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "tails": tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }

  # Parse the example (returns a dictionary of tensors)
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
  )
  return {"head": sequence_parsed["heads"], "relation": sequence_parsed["relations"], "tail": sequence_parsed["tails"], "length": context_parsed["length"]}


