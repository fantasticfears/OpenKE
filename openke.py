import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import models
import config


def build_model(model):
    """build model by config."""
    if model == "TransD":
        return models.transd
    elif model == "TransE":
        return models.transe
    elif model == 'ComplEx':
        return models.complex
    elif model == "TransH":
        pass
        # return openke.models.TransH(options)


def build_optimizer(optimizer, options):
    """build optimizer by config."""
    if optimizer == "Adagrad" or optimizer == "adagrad":
        return tf.train.AdagradOptimizer(learning_rate=options['alpha'], initial_accumulator_value=1e-8)
    elif optimizer == "Adadelta" or optimizer == "adadelta":
        return tf.train.AdadeltaOptimizer(options['alpha'])
    elif optimizer == "Adam" or optimizer == "adam":
        return tf.train.AdamOptimizer(options['alpha'])
    else:
        return tf.train.GradientDescentOptimizer(options['alpha'])


class TrainFlow(object):
    def __init__(self, step_config):
        self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")

        self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib.getHeadBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getTailBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self._config = config.TrainOptions(step_config)

        self.in_path = None
        self.out_path = None
        self.bern = 0
        self.train_times = 0
        self.margin = 1.0
        self.nbatches = 100
        self.negative_ent = 1
        self.negative_rel = 0

        self.lib.setInPath(ctypes.create_string_buffer(
            self._config.path, len(self._config.path) * 2))
        self.lib.setWorkThreads(self._config.threads)
        self.lib.importTrainFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.batch_size = self.lib.getTrainTotal() // self.nbatches
        self.sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=self._config.log_device_placement))

    def reset(self, config):
        tf.reset_default_graph()
        negativity = config.negative_entity + config.negative_relation
        triples_in_batch = 1 + negativity
        self.batch_seq_size = self.batch_size * triples_in_batch
        self.batch_h = np.zeros(
            self.batch_size * triples_in_batch, dtype=np.int64)
        self.batch_t = np.zeros(
            self.batch_size * triples_in_batch, dtype=np.int64)
        self.batch_r = np.zeros(
            self.batch_size * triples_in_batch, dtype=np.int64)
        self.batch_y = np.zeros(
            self.batch_size * triples_in_batch, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
        self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
        self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
        self.batch_y_addr = self.batch_y.__array_interface__['data'][0]

        with tf.name_scope("input"):
            self.placeholder_batch_h = tf.placeholder(
                tf.int64, [self.batch_seq_size])
            self.placeholder_batch_t = tf.placeholder(
                tf.int64, [self.batch_seq_size])
            self.placeholder_batch_r = tf.placeholder(
                tf.int64, [self.batch_seq_size])
            self.placeholder_batch_y = tf.placeholder(
                tf.float32, [self.batch_seq_size])
            self.positive_h = tf.transpose(tf.reshape(
                self.placeholder_batch_h[0:self.batch_size], [1, -1]), [1, 0])
            self.positive_t = tf.transpose(tf.reshape(
                self.placeholder_batch_t[0:self.batch_size], [1, -1]), [1, 0])
            self.positive_r = tf.transpose(tf.reshape(
                self.placeholder_batch_r[0:self.batch_size], [1, -1]), [1, 0])
            self.positive_y = tf.transpose(tf.reshape(
                self.placeholder_batch_y[0:self.batch_size], [1, -1]), [1, 0])
            self.negative_h = tf.transpose(tf.reshape(
                self.placeholder_batch_h[self.batch_size:self.batch_seq_size], [negativity, -1]), perm=[1, 0])
            self.negative_t = tf.transpose(tf.reshape(
                self.placeholder_batch_t[self.batch_size:self.batch_seq_size], [negativity, -1]), perm=[1, 0])
            self.negative_r = tf.transpose(tf.reshape(
                self.placeholder_batch_r[self.batch_size:self.batch_seq_size], [negativity, -1]), perm=[1, 0])
        self.lib.setBern(config.options['bern'])
        self.lib.randReset()

    @property
    def config(self):
        return self._config

    @property
    def entity_size(self):
        return self.entTotal

    @property
    def relation_size(self):
        return self.relTotal

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path, steps=0):
        self.exportName = path
        self.export_steps = steps

    def set_export_steps(self, steps):
        self.export_steps = steps

    def sampling(self, config):
        self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr,
                          self.batch_y_addr, self.batch_size, config.negative_entity, config.negative_relation)

    @property
    def train_feed_dict(self):
        return {
            self.placeholder_batch_h: self.batch_h,
            self.placeholder_batch_t: self.batch_t,
            self.placeholder_batch_r: self.batch_r,
            self.placeholder_batch_y: self.batch_y
        }


def import_variables(self, path=None):
    with self.graph.as_default():
        with self.sess.as_default():
            if path == None:
                self.saver.restore(self.sess, self.importName)
            else:
                self.saver.restore(self.sess, path)


def get_parameter_lists(self):
    return self.trainModel.parameter_lists


def get_parameters_by_name(self, var_name):
    with self.graph.as_default():
        with self.sess.as_default():
            if var_name in self.trainModel.parameter_lists:
                return self.sess.run(self.trainModel.parameter_lists[var_name])
            else:
                return None


def get_parameters(self, mode="numpy"):
    res = {}
    lists = self.get_parameter_lists()
    for var_name in lists:
        if mode == "numpy":
            res[var_name] = self.get_parameters_by_name(var_name)
        else:
            res[var_name] = self.get_parameters_by_name(var_name).tolist()
    return res


def save_variables_to_json(sess, variables, path):
    with open(os.path.join(path, 'variables.json'), "w") as f:
        result = {}
        for k, v in variables.iteritems():
            result[k] = sess.run(v).tolist()
        f.write(json.dumps(result))


def load_variables_from_json(sess, variables, tensor):
    with sess.as_default():
        if var_name in self.trainModel.parameter_lists:
            self.trainModel.parameter_lists[var_name].assign(
                tensor).eval()


def set_parameters(self, lists):
    for i in lists:
        self.set_parameters_by_name(i, lists[i])


def train_model_fn(config, state):
    with tf.name_scope("train"):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=True)
        with tf.variable_scope("model", reuse=False, initializer=initializer):
            model = build_model(config.model)
            variables = model.init_variables(
                state.entity_size, state.relation_size, config.options)

            opt_method = config.optimizer
            optimizer = build_optimizer(opt_method, config.options)

        loss_op = model.loss(state, variables, config.options)
        grads_and_vars = optimizer.compute_gradients(loss_op)
        for grad, var in grads_and_vars:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))
        train_op = optimizer.apply_gradients(
            grads_and_vars)

    for var in tf.trainable_variables():
        summaries.append(
            tf.summary.histogram(var.op.name, var))
    summary_op = tf.summary.merge(summaries)

    return loss_op, train_op, summary_op, variables


def _average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
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
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train_model_gpu_fn(config, state):
    with tf.name_scope("train"):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        with tf.device('/cpu:0'):
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=True)
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
                model = build_model(config.model)
                variables = model.init_variables(
                    state.entity_size, state.relation_size, config.options)

                opt_method = config.optimizer
                optimizer = build_optimizer(opt_method, config.options)

        loss_ops = []
        tower_grads = []
        for i in range(state.config.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_{0}'.format(i)) as scope:
                    loss_op = model.loss(state, variables, config.options)
                    loss_ops.append(loss_op)

                    tf.get_variable_scope().reuse_variables()

                    grads_and_vars = optimizer.compute_gradients(loss_op)
                    tower_grads.append(grads_and_vars)

        with tf.device('/cpu:0'):
            grads_and_vars = _average_gradients(tower_grads)

            for grad, var in grads_and_vars:
                if grad is not None:
                    summaries.append(tf.summary.histogram(
                        var.op.name + '/gradients', grad))
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                train_op = optimizer.apply_gradients(
                    grads_and_vars)

        for var in tf.trainable_variables():
            summaries.append(
                tf.summary.histogram(var.op.name, var))
        summary_op = tf.summary.merge(summaries)

    return loss_ops, train_op, summary_op, variables


class TestHelper(object):
    def __init__(self, path):
        self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
        self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]

        self.lib.setInPath(ctypes.create_string_buffer(
            path, len(path) * 2))
        self.lib.importTestFiles()
        self.test_h = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
        self.test_t = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
        self.test_r = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__['data'][0]
        self.test_t_addr = self.test_t.__array_interface__['data'][0]
        self.test_r_addr = self.test_r.__array_interface__['data'][0]

        self.predict_h = tf.placeholder(tf.int64, [None])
        self.predict_t = tf.placeholder(tf.int64, [None])
        self.predict_r = tf.placeholder(tf.int64, [None])

    @property
    def total_test_case(self):
        return self.lib.getTestTotal()

    def prepare_batch(self, direction='head'):
        if direction == 'head':
            self.lib.getHeadBatch(
                self.test_h_addr, self.test_t_addr, self.test_r_addr)
        else:
            self.lib.getTailBatch(
                self.test_h_addr, self.test_t_addr, self.test_r_addr)

    def test_batch(self, result, direction='head', verbose=False):
        if direction == 'head':
            self.lib.testHead(result.__array_interface__['data'][0], verbose)
        else:
            self.lib.testTail(result.__array_interface__['data'][0], verbose)

    @property
    def feed_dict(self):
        return {
            self.predict_h: self.test_h,
            self.predict_t: self.test_t,
            self.predict_r: self.test_r
        }

    def test_stat(self):
        self.lib.test()


def test_model_fn(state, model_name, options):
    with tf.name_scope("test"):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            model = build_model(model_name)
            variables = model.init_variables(
                options['entity_size'], options['relation_size'], options)
    return model.predict(state, variables)
