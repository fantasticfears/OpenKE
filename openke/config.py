"""A collection of meta class for options and convenience methods."""

import os
import yaml

def load_from_yaml(filename: str):
  """returns the parsed object from YAML file."""
  with open(filename, 'r') as config_file:
    data = yaml.load(config_file)
  return data

class TrainOptions(object):
  """A class pertains training options."""

  DEFAULT_TRAIN_CONFIG_FILENAME = 'train.yaml'

  def __init__(self, yaml_filename: str = DEFAULT_TRAIN_CONFIG_FILENAME):
    """Loads the config from a file."""
    config = load_from_yaml(yaml_filename)
    self._name = config.get('name')
    config_dir = os.path.abspath(os.path.join(yaml_filename, '..'))
    self._flow = [TrainStep(step, "{}_{}".format(self._name, idx), config_dir) for idx, step in enumerate(config.get('flow', []))]

  @property
  def flow(self):
    """gets the flow."""
    return self._flow

  @property
  def name(self):
    """gets the train plan by name."""
    return self._name

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
    self._log_device_placement = step.get('log_device_placement')
    self._num_gpus = step.get('num_gpus')
    self._optimizer = step.get('optimizer')

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
  def dataset_filename(self):
    """gets the dataset filename."""
    return self._dataset_filename

  @property
  def log_device_placement(self):
    return self._log_device_placement

  @property
  def num_gpus(self):
    return self._num_gpus

class TestOptions(object):
  """A class pertains test options."""

  DEFAULT_TEST_CONFIG_FILENAME = 'test.yaml'

  def __init__(self, yaml_filename: str = DEFAULT_TEST_CONFIG_FILENAME):
    """Loads the config from a file."""
    config = load_from_yaml(yaml_filename)
    self._path = os.path.abspath(os.path.join(yaml_filename, '..'))
    self._state_filename = config.get('state_filename')
    self._model = config.get('model')
    self._dataset_filename = config.get('dataset_filename')
    self._options = config.get('options')

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
  def dataset_filename(self):
    """gets the dataset filename."""
    return self._dataset_filename

  @property
  def path(self):
    """gets the config dir."""
    return self._path
