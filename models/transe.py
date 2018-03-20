import tensorflow as tf


def _calc(h, t, r):
    return abs(h + r - t)


def init_variables(entity_size, relation_size, options):
  """Returns needed variables for training."""
  return {
      "ent_embeddings": tf.get_variable(
          name="ent_embeddings",
          shape=[entity_size, options['embedding_size']],
          initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
      "rel_embeddings": tf.get_variable(
          name="rel_embeddings",
          shape=[relation_size, options['embedding_size']],
          initializer=tf.contrib.layers.xavier_initializer(uniform=True))
  }


def loss(state, variables, options):
    # To get positive triples and negative triples for training
    # The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
    # The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
    pos_h = state.positive_h
    pos_t = state.positive_t
    pos_r = state.positive_r
    neg_h = state.negative_h
    neg_t = state.negative_t
    neg_r = state.negative_r
    # Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
    p_h = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_h)
    p_t = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_t)
    p_r = tf.nn.embedding_lookup(variables['rel_embeddings'], pos_r)
    n_h = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_h)
    n_t = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_t)
    n_r = tf.nn.embedding_lookup(variables['rel_embeddings'], neg_r)
    # Calculating score functions for all positive triples and negative triples
    # The shape of _p_score is (batch_size, 1, hidden_size)
    # The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
    _p_score = _calc(p_h, p_t, p_r)
    _n_score = _calc(n_h, n_t, n_r)
    # The shape of p_score is (batch_size, 1)
    # The shape of n_score is (batch_size, 1)
    p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 1, keep_dims=True)
    n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 1, keep_dims=True)
    # Calculating loss to get what the framework will optimize
    return tf.reduce_sum(tf.maximum(p_score - n_score + options['margin'], 0))

def predict(state, variables):
    predict_h = state.predict_h
    predict_t = state.predict_t
    predict_r = state.predict_r
    predict_h_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_h)
    predict_t_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_t)
    predict_r_e = tf.nn.embedding_lookup(variables['rel_embeddings'], predict_r)
    return tf.reduce_mean(_calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims=False)

