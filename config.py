"""A collection of meta class for options and convenience methods."""

import os
import yaml

def load_from_yaml(filename):
  """returns the parsed object from YAML file."""
  with open(filename, 'r') as config_file:
    data = yaml.load(config_file)
  return data

class TrainOptions(object):
  """A class pertains training options."""

  DEFAULT_TRAIN_CONFIG_FILENAME = 'train.yaml'

  def __init__(self, yaml_filename=DEFAULT_TRAIN_CONFIG_FILENAME):
    """Loads the config from a file."""
    config = load_from_yaml(yaml_filename)
    self._name = config.get('name')
    config_dir = os.path.abspath(os.path.join(yaml_filename, '..')) + '/'
    self._path = config_dir
    self._num_gpus = config.get('num_gpus')
    self._flow = [TrainStep(step, "{}_{}".format(self._name, idx), config_dir) for idx, step in enumerate(config.get('flow', []))]
    self._threads = config.get('threads')
    self._log_device_placement = config.get('log_device_placement')

  @property
  def flow(self):
    """gets the flow."""
    return self._flow

  @property
  def name(self):
    """gets the train plan by name."""
    return self._name

  @property
  def num_gpus(self):
    return self._num_gpus

  @property
  def path(self):
    """gets the config dir."""
    return self._path

  @property
  def log_device_placement(self):
    return self._log_device_placement

  @property
  def threads(self):
    return self._threads


class TrainStep(object):
  """A class pertains a training step."""

  def __init__(self, step, name, path):
    """Loads the train step from a file."""
    self._state_filename = step.get('state_filename')
    self._model = step.get('model')
    self._log_level = step.get('log_level')
    self._dataset_filename = step.get('dataset_filename')
    self._options = step.get('options')
    self._name = name
    self._path = path
    self._optimizer = step.get('optimizer')
    self._nbatches = step.get('nbatches')
    self._negative_relation = step.get('negative_relation')
    self._negative_entity = step.get('negative_entity')
    self._train_iterations = step.get('train_iterations')
    self._export_path = os.path.abspath(os.path.join(self._path, step.get('export_path')))
    self._export_json = step.get('export_json')

  @property
  def export_json(self):
    return self._export_json

  @property
  def export_path(self):
    return self._export_path

  @property
  def train_iterations(self):
    return self._train_iterations

  @property
  def negative_relation(self):
    return self._negative_relation

  @property
  def negative_entity(self):
    return self._negative_entity

  @property
  def name(self):
    return self._name

  @property
  def optimizer(self):
    """gets the optimizer."""
    return self._optimizer

  @property
  def path(self):
    """gets the config dir."""
    return self._path

  @property
  def state_filename(self):
    """gets the state filename."""
    return self._state_filename

  @property
  def model(self):
    """gets the model by name."""
    return self._model

  @property
  def options(self):
    """gets the model config dictionary."""
    return self._options

  @property
  def log_level(self):
    """gets the log level."""
    return self._log_level

  @property
  def nbatches(self):
    return self._nbatches


class TestOptions(object):
  """A class pertains test options."""

  DEFAULT_TEST_CONFIG_FILENAME = 'test.yaml'

  def __init__(self, yaml_filename=DEFAULT_TEST_CONFIG_FILENAME):
    """Loads the config from a file."""
    config = load_from_yaml(yaml_filename)
    self._path = os.path.abspath(os.path.join(yaml_filename, '..'))
    self._model = config.get('model')
    self._options = config.get('options')
    self._triplet_list_filename = config.get('triplet_list_filename')
    self._entity_map_filename = config.get('entity_map_filename')
    self._relation_map_filename = config.get('relation_map_filename')

  @property
  def model(self):
    """gets the model by name."""
    return self._model

  @property
  def options(self):
    """gets the model config dictionary."""
    return self._options

  @property
  def path(self):
    """gets the config dir."""
    return self._path

  @property
  def triplet_list_filename(self):
    return self._triplet_list_filename

  @property
  def entity_map_filename(self):
    return self._entity_map_filename

  @property
  def relation_map_filename(self):
    return self._relation_map_filename
