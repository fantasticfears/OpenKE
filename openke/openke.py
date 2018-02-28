import numpy as np
import tensorflow as tf
import os
import time
import datetime
from functools import reduce
from openke.datasetutils import parse_triplets_from_sequence_example

from openke.config import TrainStep, TrainOptions
import openke.models

MOVING_AVERAGE_DECAY = 0.9999

def _parse_function(example_proto):
  features = {"h": tf.FixedLenFeature((), tf.int64),
              "r": tf.FixedLenFeature((), tf.int64),
              "t": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["h"], parsed_features["r"], parsed_features["t"]

def build_model(model, options):
  """build model by config."""
  if model == "TransD":
    pass
    # return openke.models.TransD(options)
  elif model == "TransE":
    return openke.models.transe
  elif model == "TransH":
    pass
    # return openke.models.TransH(options)

def build_optimizer(optimizer, options):
  """build optimizer by config."""
  if optimizer == "Adagrad" or optimizer == "adagrad":
    return tf.train.AdagradOptimizer(learning_rate = options['alpha'], initial_accumulator_value=1e-8)
  elif optimizer == "Adadelta" or optimizer == "adadelta":
    return tf.train.AdadeltaOptimizer(options['alpha'])
  elif optimizer == "Adam" or optimizer == "adam":
    return tf.train.AdamOptimizer(options['alpha'])
  else:
    return tf.train.GradientDescentOptimizer(options['alpha'])

def prepare_batch(next_element):
  # el = [ for n in next_elements]
  # el = reduce(lambda x,y: x+y, el)
  # print(next_element[0].shape)
  tensors = []
  for element in next_element:
    tensors.append(tf.convert_to_tensor(element[0]))
  # print(session.run(tf.Print(tensors, [tensors])))
  return tensors

class Step(object):
  """A class takes a train config and execute a run."""

  def __init__(self, config: TrainStep):
    self._config = config
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=self._config.log_device_placement)
    self._session = tf.Session(config=config)
    self._saver = None
    self._optimizer = None
    self._model = None
    self._dataset = None
    self._iterator = None
    self._next_element = None
    self._staging_area = None
    self._global_step = None
    self._model_variables = None

  @property
  def config(self):
    """gets the train config."""
    return self._config

  def initialize(self):
    """setup the model and configurations."""
    with self._session.as_default():
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      with tf.variable_scope("model", reuse=None, initializer=initializer):
        self._model = build_model(self._config.model, self._config.options)
        self._optimizer = build_optimizer(self._config.optimizer, self._config.options)
        self._model_variables = self._model.init_variables(self._config.options)

      self._global_step = tf.Variable(0, name="global_step", trainable=False)
      # self._session.run(tf.global_variables_initializer())

      train_iterations = self._config.options['train_iterations']
      self._dataset = tf.data.TFRecordDataset(self._config.dataset_filename)
      self._dataset = self._dataset.map(parse_triplets_from_sequence_example)
      self._dataset = self._dataset.padded_batch(1, padded_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([None])
      ))
      self._dataset = self._dataset.repeat(train_iterations)

      self._iterator = self._dataset.make_initializable_iterator()
      self._next_element = self._iterator.get_next()
      self._staging_area = tf.contrib.staging.StagingArea(dtypes=[tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64])

      # warm up
      self._session.run(self._iterator.initializer)
      # batch = prepare_batch(self._next_elements, self._session)
      # print(batch.dtype, self._session.run(tf.Print(batch, [batch])))
      # self._staging_area.put((batch,))

  def average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)

      average_grads.append(grad_and_var)
    return average_grads

  def save_progress(self, step):
    checkpoint_path = os.path.join(self._config.path, self._config.state_filename)
    self._saver.save(self._session, checkpoint_path, global_step=step)

  def run(self, restore_if_available=True):
    """train next batch."""

    losses = []
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      # for i in range(self._config.num_gpus):
        # with tf.device('/gpu:%d' % i):
      with tf.name_scope('%s_%d' % (self._config.name, 0)) as scope:
        # batch_data = prepare_batch(self._next_element)
        loss = self._model.loss(scope, self._next_element, self._model_variables, self._config.options, self._session)

        tf.get_variable_scope().reuse_variables()
        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        grad = self._optimizer.compute_gradients(loss)
        print("-"*40, grad)
        tower_grads.append(grad)
        losses.append(loss)

    total_loss_op = tf.add_n(losses)
    grads = tower_grads[0]
    # grads = self.average_gradients(tower_grads)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = self._optimizer.apply_gradients(grads, global_step=self._global_step)
    summaries.append(tf.summary.scalar('global_step', self._global_step))

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY, self._global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    # train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    self._saver = tf.train.Saver(tf.global_variables())
    if restore_if_available and (
      self._config.state_filename is not None and os.path.exists(self._config.state_filename)):
      self._saver.restore(self._session, self._config.state_filename)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    self._session.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=self._session)

    summary_writer = tf.summary.FileWriter(self._config.path, self._session.graph)

    step = 0
    result = 0.0
    while True:
      step += 1
      if step % 100 == 0:
        result = 0.0

      start_time = time.time()

      try:
        _, loss_value = self._session.run([apply_gradient_op, total_loss_op])
      except tf.errors.OutOfRangeError as e:
        self.save_progress(step)
        return

      result += loss_value
      duration = time.time() - start_time

      if step % 10 == 0:
        num_examples_per_step = 512
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / 1 #self._config.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.datetime.now(), step, result,
                            examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0:
        self.save_progress(step)

class TrainFlow(object):
  """Stores and executes a flow of training models."""

  def __init__(self, train_options: TrainOptions):
    """Init the train flow based on train options."""
    self._config = train_options

  def train(self):
    """Train model based on the flow."""
    for step_config in self._config.flow:
      step = Step(step_config)
      step.initialize()
      step.run()

class Reasonator(object):
  """Applies link prediction and triplet classification."""
  pass
