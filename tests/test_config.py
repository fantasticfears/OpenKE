import unittest
from testfixtures import TempDirectory
import os
from context import openke

class TestLoadingConfigFile(unittest.TestCase):
  def setUp(self):
    self._d = TempDirectory()
    self._d.write('train.yaml', b"""
model_options:
  general: &general
    embedding_size:
      entity: 100
      relation: 100
    train_iterations: 1000
    batch_size: 256
    negative_example:
      entity: True
      relaiton: False
    margin: 1.0
    alpha: 0.001
    lambda: 0.000
    optimizer: SGD
  transE: &transE
    <<: *general

name: "simple_transe"
flow:
  - model: TransE
    log_level: VERBOSE
    state_filename: "trans_e"
    dataset_filenames:
      train:
        - ""
      negative_train:
        -
    options:
      <<: *transE
""")

  def tearDown(self):
    self._d.cleanup()

  def test_general(self):
    config = openke.config.load_from_yaml(os.path.join(self._d.path, 'train.yaml'))
    self.assertEqual(config['name'], 'simple_transe')
    step_0 = config['flow'][0]
    self.assertEqual(step_0['model'], 'TransE')
    self.assertEqual(step_0['state_filename'], 'trans_e')
    self.assertEqual(step_0['options']['train_iterations'], 1000)

class TestConfigMethods(unittest.TestCase):
  def setUp(self):
    self._d = TempDirectory()
    self._d.write('train.yaml', b"""
model_options:
  general: &general
    embedding_size:
      entity: 100
      relation: 100
    train_iterations: 1000
    batch_size: 256
    negative_example:
      entity: True
      relaiton: False
    margin: 1.0
    alpha: 0.001
    lambda: 0.000
    optimizer: SGD
  transE: &transE
    <<: *general

name: "simple_transe"
flow:
  - model: TransE
    log_level: VERBOSE
    state_filename: "trans_e"
    dataset_filenames:
      train:
        - ""
      negative_train:
        -
    options:
      <<: *transE
""")
    self._config_options = openke.config.TrainOptions(os.path.join(self._d.path, 'train.yaml'))

  def tearDown(self):
    self._d.cleanup()

  def test_train_name(self):
    self.assertEqual(self._config_options.name, 'simple_transe')

  def test_train_flow_model(self):
    self.assertEqual(self._config_options.flow[0].model, 'TransE')

  def test_train_flow_state_filename(self):
    self.assertEqual(self._config_options.flow[0].state_filename, 'trans_e')

  def test_train_flow_options(self):
    self.assertEqual(self._config_options.flow[0].options['train_iterations'], 1000)

if __name__ == '__main__':
  unittest.main()
