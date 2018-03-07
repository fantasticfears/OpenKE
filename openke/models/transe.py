import tensorflow as tf

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

def loss(train_data, variables, options):
  #To get positive triples and negative triples for training
  #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
  #The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
  pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = train_data

  p_h = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_h)
  p_t = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_t)
  p_r = tf.nn.embedding_lookup(variables['rel_embeddings'], pos_r)
  n_h = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_h)
  n_t = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_t)
  n_r = tf.nn.embedding_lookup(variables['rel_embeddings'], neg_r)

  #The shape of _p_score is (batch_size, 1, hidden_size)
  #The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
  _p_score = _calc(p_h, p_r, p_t)
  _n_score = _calc(n_h, n_r, n_t)
  p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 1, keep_dims=True)
  n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 1, keep_dims=True)
  return tf.reduce_sum(tf.maximum(p_score - n_score + options['margin'], 0))

def predict(test_data, variables, options):
  predict_h, predict_r, predict_t = test_data

  predict_h_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_h)
  predict_t_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_t)
  predict_r_e = tf.nn.embedding_lookup(variables['rel_embeddings'], predict_r)
  return tf.reduce_sum(_calc(predict_h_e, predict_t_e, predict_r_e), 2, keep_dims=False)
