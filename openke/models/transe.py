import tensorflow as tf
import openke.models

def _calc(h, r, t):
  return abs(h + r - t)

def init_variables(options):
  """Returns needed variables for training."""
  return {
    "ent_embeddings": tf.get_variable(
      name="ent_embeddings",
      shape=[options['embedding_size']['entity'], options['embedding_size']['hidden']],
      initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    "rel_embeddings": tf.get_variable(
      name="rel_embeddings",
      shape=[options['embedding_size']['relation'], options['embedding_size']['hidden']],
      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
  }

def loss(scope, train_data, variables, options, sess):
  #To get positive triples and negative triples for training
  #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
  #The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
  pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = train_data
  #Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
  p_h = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_h)
  p_t = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_t)
  p_r = tf.nn.embedding_lookup(variables['rel_embeddings'], pos_r)
  n_h = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_h)
  n_t = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_t)
  n_r = tf.nn.embedding_lookup(variables['rel_embeddings'], neg_r)
  #Calculating score functions for all positive triples and negative triples
  #The shape of _p_score is (batch_size, 1, hidden_size)
  #The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
  # print(tf.shape(pos_h), tf.shape(pos_r), tf.shape(pos_t))
  # print(tf.shape(p_h), tf.shape(p_r), tf.shape(p_t))
  _p_score = _calc(p_h, p_r, p_t)
  _n_score = _calc(n_h, n_r, n_t)
  #The shape of p_score is (batch_size, 1)
  #The shape of n_score is (batch_size, 1)
  # print(_p_score, sess.run(tf.Print(_p_score, [_p_score])))
  p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 0, keep_dims=True)
  n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 0, keep_dims=True)
  #Calculating loss to get what the framework will optimize
  return tf.reduce_sum(tf.maximum(p_score - n_score + options['margin'], 0))
