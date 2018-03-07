import tensorflow as tf

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
          initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
      "ent_transfer": tf.get_variable(
          name="ent_transfer",
          shape=[options['embedding_size']['entity'], options['embedding_size']['hidden']],
          initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
		  "rel_transfer": tf.get_variable(
          name="rel_transfer",
          shape=[options['embedding_size']['relation'], options['embedding_size']['hidden']],
          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
  }

def _transfer(e, t, r):
  return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

def _calc(h, t, r):
  return abs(h + r - t)

def loss(train_data, variables, options):
  #To get positive triples and negative triples for training
  #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
  #The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
  pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = train_data
  #Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
  pos_h_e = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_h)
  pos_t_e = tf.nn.embedding_lookup(variables['ent_embeddings'], pos_t)
  pos_r_e = tf.nn.embedding_lookup(variables['rel_embeddings'], pos_r)
  neg_h_e = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_h)
  neg_t_e = tf.nn.embedding_lookup(variables['ent_embeddings'], neg_t)
  neg_r_e = tf.nn.embedding_lookup(variables['rel_embeddings'], neg_r)
  #Getting the required parameters to transfer entity embeddings, e.g. pos_h_t, pos_t_t and pos_r_t are transfer parameters for positive triples
  pos_h_t = tf.nn.embedding_lookup(variables['ent_transfer'], pos_h)
  pos_t_t = tf.nn.embedding_lookup(variables['ent_transfer'], pos_t)
  pos_r_t = tf.nn.embedding_lookup(variables['rel_transfer'], pos_r)
  neg_h_t = tf.nn.embedding_lookup(variables['ent_transfer'], neg_h)
  neg_t_t = tf.nn.embedding_lookup(variables['ent_transfer'], neg_t)
  neg_r_t = tf.nn.embedding_lookup(variables['rel_transfer'], neg_r)
  #Calculating score functions for all positive triples and negative triples
  p_h = _transfer(pos_h_e, pos_h_t, pos_r_t)
  p_t = _transfer(pos_t_e, pos_t_t, pos_r_t)
  p_r = pos_r_e
  n_h = _transfer(neg_h_e, neg_h_t, neg_r_t)
  n_t = _transfer(neg_t_e, neg_t_t, neg_r_t)
  n_r = neg_r_e
  #The shape of _p_score is (batch_size, 1, hidden_size)
  #The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
  _p_score = _calc(p_h, p_t, p_r)
  _n_score = _calc(n_h, n_t, n_r)
  #The shape of p_score is (batch_size, 1)
  #The shape of n_score is (batch_size, 1)
  p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 1, keep_dims=True)
  n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 1, keep_dims=True)
  #Calculating loss to get what the framework will optimize
  return tf.reduce_sum(tf.maximum(p_score - n_score + options['margin'], 0))

def predict(test_data, variables, options):
  predict_h, predict_r, predict_t = test_data
  predict_h_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_h)
  predict_t_e = tf.nn.embedding_lookup(variables['ent_embeddings'], predict_t)
  predict_r_e = tf.nn.embedding_lookup(variables['rel_embeddings'], predict_r)
  predict_h_t = tf.nn.embedding_lookup(variables['ent_transfer'], predict_h)
  predict_t_t = tf.nn.embedding_lookup(variables['ent_transfer'], predict_t)
  predict_r_t = tf.nn.embedding_lookup(variables['rel_transfer'], predict_r)
  h_e = _transfer(predict_h_e, predict_h_t, predict_r_t)
  t_e = _transfer(predict_t_e, predict_t_t, predict_r_t)
  r_e = predict_r_e
  return tf.reduce_sum(_calc(h_e, t_e, r_e), 2, keep_dims=True)
