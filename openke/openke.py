import numpy as np
import tensorflow as tf
import os
import time
import datetime

from openke.config import TrainStep, TrainOptions
import openke.models

MOVING_AVERAGE_DECAY = 0.9999

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

class Step(object):
  """A class takes a train config and execute a run."""

  def __init__(self, config: TrainStep):
    self._config = config
    self._graph = tf.get_default_graph()
    self._session = tf.Session(graph=self._graph)
    self._saver = tf.train.Saver()
    self._optimizer = None
    self._model = None
    self._dataset = None
    self._iterator = None
    self._next_element = None
    self._stage_area = None
    self._global_step = None
    self._model_variables = None

  @property
  def config(self):
    """gets the train config."""
    return self._config

  def initialize(self, restore_if_available=True):
    """setup the model and configurations."""
    with self._session.as_default():
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      with tf.variable_scope("model", reuse=None, initializer=initializer):
        self._model = build_model(self._config.model, self._config.options)
        self._optimizer = build_optimizer(self._config.optimizer, self._config.options)

      self._model_variables = self._model.init_variables(self._config.options)
      self._global_step = tf.Variable(0, name="global_step", trainable=False)
      self._session.run(tf.initialize_all_variables())

      if restore_if_available and self._config.state_filename is not None:
        self._saver.restore(self._session, self._config.state_filename)

      filenames = (self._config.dataset_filenames['train'],
                   self._config.dataset_filenames['negative_train'])
      self._dataset = tf.data.TFRecordDataset.zip(filenames)
      self._dataset = self._dataset.batch(self._config.options['batch_size'])
      self._dataset = self._dataset.repeat(self._config.options['train_iterations'])
      self._iterator = self._dataset.make_initializable_iterator()
      self._next_element = self._iterator.get_next()
      self._stage_area = tf.contrib.staging.StagingArea()

      # warm up
      self._stage_area.put(self._session.run(self._next_element))

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

  def run(self):
    """train next batch."""

    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(self._config.num_gpus):
        try:
          self._stage_area.put(self._session.run(self._next_element))
        except tf.errors.OutOfRangeError:
          pass
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (self._config.name, i)) as scope:
            loss = self._model.loss(scope, self._stage_area.get(), self._model_variables, self._config.options)

            tf.get_variable_scope().reuse_variables()
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            grads = self._optimizer.compute_gradients(loss)
            tower_grads.append(grads)

    grads = self.average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', grads))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = self._optimizer.apply_gradients(grads, global_step=self._global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, self._global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    self._saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=self._config.log_device_placement))
    self._session.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(self._config.path, self._session.graph)

    for step in range(self._config.options['train_iterations']):
      start_time = time.time()
      _, loss_value = self._session.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = self._config.options['batch_size'] * self._config.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / self._config.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == self._config.options['train_iterations']:
        checkpoint_path = os.path.join(self._config.path, self._config.state_filename)
        self._saver.save(self._session, checkpoint_path, global_step=step)

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
