"""Imports data to TFRecord, like FB15K."""

from dataset_utils import dataset_exists, convert_dataset, write_mapping_file
from typing import Tuple, List
import tensorflow as tf
import os.path
import csv

flags = tf.flags
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')
flags.DEFINE_string('dataset_file', None, 'String: Your dataset file')
flags.DEFINE_string('split_name', 'train', '`train` or `validation`')
flags.DEFINE_integer('num_shards', 1, 'Int: Number of shards to split the TFRecord files')
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def _get_entities_and_relations_from_tsv(dataset_dir: str, filename: str) -> Tuple[List, List, int]:
  """gets entities and relations list and LOC from a tsv file containing triplets."""
  relations = []
  entities = []
  loc = 0
  with open(os.path.join(dataset_dir, filename), 'r') as f:
    tsv = csv.reader(f, delimiter='\t')
    for row in tsv:
      entities.append(row[0])
      entities.append(row[2])
      relations.append(row[1])
      loc += 1

  return list(set(entities)), list(set(relations)), loc

def main():
  if not FLAGS.tfrecord_filename:
    raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

  if not FLAGS.dataset_dir:
    raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

  if dataset_exists(dataset_dir=FLAGS.dataset_dir,
                    num_shards=FLAGS.num_shards,
                    output_filename=FLAGS.tfrecord_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return None

  entities, relations, loc = _get_entities_and_relations_from_tsv(FLAGS.dataset_dir, FLAGS.dataset_file)

  entities_to_ids = dict(zip(entities, range(len(entities))))
  relations_to_ids = dict(zip(relations, range(len(relations))))

  with open(os.path.join(FLAGS.dataset_dir, FLAGS.dataset_file), 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    convert_dataset(FLAGS.split_name, loc,
                    reader, 1, 1, True,
                    entities_to_ids, relations_to_ids,
                    FLAGS.dataset_dir, FLAGS.tfrecord_filename, FLAGS.num_shards)

  write_mapping_file(entities_to_ids, 'train2id.txt', FLAGS.dataset_dir)
  write_mapping_file(relations_to_ids, 'relation2id.txt', FLAGS.dataset_dir)

  print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
  main()
