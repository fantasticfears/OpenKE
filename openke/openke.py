import numpy as np
import tensorflow as tf
import os
import time
import datetime
from functools import reduce
from openke.datasetutils import parse_triplets_from_sequence_example, parse_test_triplets_from_sequence_example, read_training_triplets_file

from openke.config import TrainStep, TrainOptions, TestOptions
import openke.models

def build_model(model):
  """build model by config."""
  if model == "TransD":
    return openke.models.transd
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

def prepare_batch(next_element, options):
  pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = next_element
  pos_h = tf.transpose(pos_h)
  pos_r = tf.transpose(pos_r)
  pos_t = tf.transpose(pos_t)
  neg_h = tf.reshape(neg_h, [options['batch_size'], -1])
  neg_r = tf.reshape(neg_r, [options['batch_size'], -1])
  neg_t = tf.reshape(neg_t, [options['batch_size'], -1])
  return pos_h, pos_r, pos_t, neg_h, neg_r, neg_t

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

  def initialize(self, global_step):
    """setup the model and configurations."""
    print("-" * 80, "\nInitializing " + self._config.name)
    with self._session.as_default():
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
        self._model = build_model(self._config.model)
        self._optimizer = build_optimizer(self._config.optimizer, self._config.options)
        self._model_variables = self._model.init_variables(self._config.options)

      self._global_step = tf.Variable(global_step, name="global_step", trainable=False)

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
      # self._staging_area = tf.contrib.staging.StagingArea(dtypes=[tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64])

      self._session.run(self._iterator.initializer)

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
      for i in range(self._config.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (self._config.name, i)) as scope:
            loss = self._model.loss(prepare_batch(self._next_element, self._config.options), self._model_variables, self._config.options)

            tf.get_variable_scope().reuse_variables()
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            grad = self._optimizer.compute_gradients(loss)
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

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    self._session.run(init)

    self._saver = tf.train.Saver(self._model_variables)
    if restore_if_available and (
      self._config.state_filename is not None and os.path.exists(self._config.state_filename)):
      self._saver.restore(self._session, self._config.state_filename)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=self._session)

    summary_writer = tf.summary.FileWriter(self._config.path, self._session.graph)

    step = 0
    result = 0.0
    print("-" * 80, "\Running " + self._config.name)
    while True:
      step += 1
      if step % 100 == 0:
        result = 0.0

      start_time = time.time()

      try:
        if step % 1000 == 0:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          _, loss_value = self._session.run([apply_gradient_op, total_loss_op],
            options=run_options,
            run_metadata=run_metadata)
          summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
        else:
          _, loss_value = self._session.run([apply_gradient_op, total_loss_op])
      except tf.errors.OutOfRangeError as e:
        self.save_progress(self._global_step)
        return tf.train.global_step(self._session, self._global_step)

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
        self.save_progress(self._global_step)

class TrainFlow(object):
  """Stores and executes a flow of training models."""

  def __init__(self, train_options: TrainOptions):
    """Init the train flow based on train options."""
    self._config = train_options

  def train(self):
    """Train model based on the flow."""
    global_step = 0
    for step_config in self._config.flow:
      step = Step(step_config)
      step.initialize(global_step)
      global_step = step.run()

class Reasonator(object):
  """Applies link prediction and triplet classification."""
  def __init__(self, state_path, model, entity_map_path, relation_map_path):
    self._config = config
    self._session = tf.Session()
    self._saver = None
    self._model = None
    self._model_variables = None

    self._state_path = state_path
    self._entity_map_path = entity_map_path
    self._relation_map_path = relation_map_path
    self._session = tf.Session()

    with self._session.as_default():
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      with tf.variable_scope("model", reuse=True, initializer=initializer):
        self._model = build_model(self._config.model)
        self._model_variables = self._model.init_variables(self._config.options)

    self._saver = tf.train.Saver(self._model_variables)
    self._saver.restore(self._session, self._config.state_filename)

  def classify(triplet):
    self._session.run([self._model.predict(triplet, self._model_variables, self._config.options)])

  def predict(triplet, direction=0):
    pass
    # self._session.run


class EmbeddingsTest(object):
  def __init__(self, config):
    self._config = config
    self._session = tf.Session()

    self._l_filter_total = 0
    self._l_total = 0
    self._l3_filter_total = 0
    self._l3_total = 0
    self._l1_filter_total = 0
    self._l_filter_rank = 0
    self._l_rank = 0

    self._r_filter_total = 0
    self._r_total = 0
    self._r3_filter_total = 0
    self._r3_total = 0
    self._r1_filter_total = 0
    self._r_filter_rank = 0
    self._r_rank = 0

    self._triplet_list = read_training_triplets_file(self._config.path, self._config.triplet_list_filename)
    self._total_triplet = len(self._triplet_list)
    self._total_entity = self._config.options['embedding_size']['entity']

    with self._session.as_default():
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      with tf.variable_scope("model", reuse=False, initializer=initializer):
        self._model = build_model(self._config.model)
        self._model_variables = self._model.init_variables(self._config.options)

      self._global_step = tf.Variable(0, name="global_step", trainable=False)

      self._dataset = tf.data.TFRecordDataset(self._config.dataset_filename)
      self._dataset = self._dataset.map(parse_test_triplets_from_sequence_example)
      # self._dataset = self._dataset.padded_batch(1, padded_shapes=(
      #   tf.TensorShape([None]),
      #   tf.TensorShape([None]),
      #   tf.TensorShape([None]),
      #   tf.TensorShape([None])
      # ))
      self._iterator = self._dataset.make_initializable_iterator()
      self._next_element = self._iterator.get_next()

      self._session.run(self._iterator.initializer)

    init = tf.global_variables_initializer()
    self._session.run(init)

    self._saver = tf.train.Saver(self._model_variables)
    self._saver.restore(self._session, self._config.state_filename)

  def _prepare_test_batch(self, next_element, batch_size):
    h, r, t, type_ = next_element

    h = tf.expand_dims(h, 1) # (batch_size, 1)
    r = tf.expand_dims(r, 1) # (batch_size, 1)
    t = tf.expand_dims(t, 1) # (batch_size, 1)
    triplets = tf.concat([h, r, t], axis=1)

    r_s = tf.split(tf.tile(r, [1, self._total_entity]), batch_size, axis=0) # [(batch_size, 1)*_total_entity]
    r_s = [tf.transpose(r) for r in r_s]
    range_ents = tf.expand_dims(tf.range(0, self._total_entity), 1) # (total_entity, 1)
    if type_ == b'head':
      h_s = [range_ents] * batch_size
      t_s = tf.split(tf.tile(t, [1, self._total_entity]), batch_size, axis=0) # [(1, _total_entity)] * batch_size
      t_s = [tf.transpose(t) for t in t_s]
    else:
      h_s = tf.split(tf.tile(h, [1, self._total_entity]), batch_size, axis=0) # [(batch_size, 1)*_total_entity]
      h_s = [tf.transpose(h) for h in h_s]
      t_s = [range_ents] * batch_size

    batch = []
    # h_s = tf.Print(h_s, [h_s])
    # r_s = tf.Print(r_s, [r_s])
    # t_s = tf.Print(t_s, [t_s])
    for h, r, t in zip(h_s, r_s, t_s):
      batch.append((h, r, t))

    return batch, type_, triplets

  def run(self):
    test_data, type_op, triplets_op = self._prepare_test_batch(self._next_element, self._config.options['batch_size'])
    predicts = [self._model.predict(d, self._model_variables, self._config.options) for d in test_data]
    while True:
      try:
        predicts_result, type_, triplets = self._session.run([predicts, type_op, triplets_op])
      except tf.errors.OutOfRangeError as e:
        return self._report()
      for result, triplet in zip(predicts_result, triplets):
        self._add_to_stat(result, type_, triplet)

  def _add_to_stat(self, result, type_, triplet):
    """Process result tensor."""
    if type_ == b'head':
      self._test_head(result, triplet)
    else:
      self._test_tail(result, triplet)

  def _test_head(self, result, triplet):
    h, t, r = triplet
    predicted = result[h]
    print(len(result), predicted)

    l_pos = 0
    l_filter_pos = 0

    for ent in range(self._total_entity):
      val = result[ent]
      if val < predicted:
        l_pos += 1
        if not self._find(ent, r, t):
          l_filter_pos += 1

    if l_filter_pos < 10:
      self._l_filter_total += 1
    if l_pos < 10:
      self._l_total += 1
    if l_filter_pos < 3:
      self._l3_filter_total += 1
    if l_pos < 3:
      self._l3_total += 1
    if l_filter_pos < 1:
      self._l1_filter_total += 1

    self._l_filter_rank += l_filter_pos + 1
    self._l_rank += 1 + l_pos

  def _test_tail(self, result, triplet):
    h, t, r = triplet
    predicted = result[h]

    r_pos = 0
    r_filter_pos = 0

    for ent in range(self._total_entity):
      val = result[ent]
      if val < predicted:
        r_pos += 1
        if not self._find(h, r, ent):
          r_filter_pos += 1

    if r_filter_pos < 10:
      self._r_filter_total += 1
    if r_pos < 10:
      self._r_total += 1
    if r_filter_pos < 3:
      self._r3_filter_total += 1
    if r_pos < 3:
      self._r3_total += 1
    if r_filter_pos < 1:
      self._r1_filter_total += 1

    self._r_filter_rank += r_filter_pos + 1
    self._r_rank += 1 + r_pos

  def _find(self, h, r, t):
    lef = 0
    rig = self._total_triplet - 1
    mid = 0
    while lef + 1 < rig:
      mid = (lef + rig) // 2
      t = self._triplet_list[mid]
      if t[0] < h or (t[0] == h and t[1] < r) or ((t[0], t[1]) == (h, r) and t[2] < t):
        lef = mid
      else:
        rig = mid

    l_t = self._triplet_list[lef]
    if l_t == (h, r, t):
      return True
    r_t = self._triplet_list[rig]
    if r_t == (h, r, t):
      return True
    return False

  def _report(self):
    print(self._l_filter_total, self._l_total, self._l3_filter_total, self._l3_total, self._l1_filter_total, self._l_filter_rank, self._l_rank)
    print(self._r_filter_total, self._r_total, self._r3_filter_total, self._r3_total, self._r1_filter_total, self._r_filter_rank, self._r_rank)
