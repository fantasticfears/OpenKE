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

def write_training_triplets_file(triplets, filename: str, dataset_dir: str):
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as label_file:
    for h, r, t, _ in triplets:
      label_file.write('%d:%d:%d\n' % (h, r, t))

def read_training_triplets_file(dataset_dir, filename):
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'r') as file:
    lines = file.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  triplets = []
  for line in lines:
    triplets.append(tuple(line.split(':')))
  return triplets

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
  write_mapping_file(entities_to_ids, 'entity_id.map', dataset_dir)
  write_mapping_file(relations_to_ids, 'relation_id.map', dataset_dir)

  gentrain.init_buff(num_triplets, len(entities_to_ids), len(relations_to_ids))

  for head, relation, tail in reader:
    triplet = (entities_to_ids[head],
                relations_to_ids[relation],
                entities_to_ids[tail])
    read_triplet = gentrain.feed(triplet)
    sys.stdout.write('\r>> Read triplet %d' % (read_triplet))
    sys.stdout.flush()

  gentrain.freq()

  training_triplets = []
  sys.stdout.write('\r>> Proceeding to generate training triplets')
  sys.stdout.flush()
  with tf.python_io.TFRecordWriter("train_triplets.tfrecord") as writer:
    max_shard = read_triplet // batch_size + 1
    for shard in range(max_shard):
      triplets = []
      for _ in range(batch_size):
        sys.stdout.write('\r>> Writing triplet shard %d/%d' % (
            shard, max_shard))

        t = gentrain.yield_triplets(negative_relation_rate, negative_entity_rate, bern)
        triplets.append(t)
        training_triplets.append(t[0])

      ex = triplets_to_tf_example(triplets)
      writer.write(ex.SerializeToString())

  write_training_triplets_file(training_triplets, 'training_triplets.list', dataset_dir)

  sys.stdout.write('\r>> Proceeding to generate test triplets')
  sys.stdout.flush()
  max_entity = len(entities_to_ids)
  with tf.python_io.TFRecordWriter("test_triplets.tfrecord") as writer:
    max_shard = read_triplet // 20 // batch_size + 1
    for type_ in ["head", "tail"]:
      for shard in range(max_shard):
        triplets = []

        for _ in range(batch_size):
          sys.stdout.write('\r>> Writing triplet shard %d/%d' % (
              shard, max_shard))

          triplets.append(gentrain.yield_triplets(0, 0, False))

        ex = test_triplets_to_tf_example(triplets, type_)
        writer.write(ex.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def test_triplets_to_tf_example(batch, type_):
  ex = tf.train.SequenceExample()
  # context for sequence example
  ex.context.feature["type"].bytes_list.value.append(type_.encode())
  triplets_length = len(batch) * len(batch[0])
  ex.context.feature["length"].int64_list.value.append(triplets_length)

  # Feature lists for sequential features of our example
  head_tokens = ex.feature_lists.feature_list["heads"]
  relation_tokens = ex.feature_lists.feature_list["relations"]
  tail_tokens = ex.feature_lists.feature_list["tails"]

  for triplets in batch:
    for h, r, t, _ in triplets:
      head_tokens.feature.add().int64_list.value.append(h)
      relation_tokens.feature.add().int64_list.value.append(r)
      tail_tokens.feature.add().int64_list.value.append(t)

  return ex

def triplets_to_tf_example(batch):
  ex = tf.train.SequenceExample()
  # context for sequence example
  triplets_length = len(batch) * len(batch[0])
  ex.context.feature["length"].int64_list.value.append(triplets_length)

  # Feature lists for sequential features of our example
  head_tokens = ex.feature_lists.feature_list["heads"]
  relation_tokens = ex.feature_lists.feature_list["relations"]
  tail_tokens = ex.feature_lists.feature_list["tails"]
  neg_head_tokens = ex.feature_lists.feature_list["negative_heads"]
  neg_relation_tokens = ex.feature_lists.feature_list["negative_relations"]
  neg_tail_tokens = ex.feature_lists.feature_list["negative_tails"]

  for triplets in batch:
    for h, r, t, neg_flag in triplets:
      if neg_flag > 0:
        head_tokens.feature.add().int64_list.value.append(h)
        relation_tokens.feature.add().int64_list.value.append(r)
        tail_tokens.feature.add().int64_list.value.append(t)
      else:
        neg_head_tokens.feature.add().int64_list.value.append(h)
        neg_relation_tokens.feature.add().int64_list.value.append(r)
        neg_tail_tokens.feature.add().int64_list.value.append(t)

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
    "tails": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "negative_heads": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "negative_relations": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "negative_tails": tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }

  # Parse the example (returns a dictionary of tensors)
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
  )
  return (sequence_parsed["heads"],
      sequence_parsed["relations"],
      sequence_parsed["tails"],
      sequence_parsed["negative_heads"],
      sequence_parsed["negative_relations"],
      sequence_parsed["negative_tails"]
  )

def parse_test_triplets_from_sequence_example(ex):
  '''
  Explain to TF how to go froma  serialized example back to tensors
  :param ex:
  :return: A dictionary of tensors, in this case {seq: The sequence, length: The length of the sequence}
  '''
  context_features = {
    "type": tf.FixedLenFeature([], dtype=tf.string),
    "length": tf.FixedLenFeature([], dtype=tf.int64),
    "triplet": tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    "heads": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "relations": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "tails": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  }

  # Parse the example (returns a dictionary of tensors)
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
  )
  return (sequence_parsed["heads"],
      sequence_parsed["relations"],
      sequence_parsed["tails"],
      context_features["type"],
      context_features["triplet"]
  )
