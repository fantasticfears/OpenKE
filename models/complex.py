import tensorflow as tf


def init_variables(entity_size, relation_size, options):
    ent_re_embeddings = tf.get_variable(name="ent_re_embeddings", shape=[
        entity_size, options['embedding_size']], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    rel_re_embeddings = tf.get_variable(name="rel_re_embeddings", shape=[
        relation_size, options['embedding_size']], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    ent_im_embeddings = tf.get_variable(name="ent_im_embeddings", shape=[
        entity_size, options['embedding_size']], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    rel_im_embeddings = tf.get_variable(name="rel_im_embeddings", shape=[
        relation_size, options['embedding_size']], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    return {"ent_re_embeddings": ent_re_embeddings,
            "ent_im_embeddings": ent_im_embeddings,
            "rel_re_embeddings": rel_re_embeddings,
            "rel_im_embeddings": rel_im_embeddings}


def _calc(e1_h, e2_h, e1_t, e2_t, r1, r2):
    return e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2


def loss(state, variables, options):
    # Obtaining the initial configuration of the model

    h = state.placeholder_batch_h
    t = state.placeholder_batch_t
    r = state.placeholder_batch_r
    # To get positive triples and negative triples for training
    # To get labels for the triples, positive triples as 1 and negative triples as -1
    # The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
    y = state.placeholder_batch_y
    # Embedding entities and relations of triples
    e1_h = tf.nn.embedding_lookup(variables['ent_re_embeddings'], h)
    e2_h = tf.nn.embedding_lookup(variables['ent_im_embeddings'], h)
    e1_t = tf.nn.embedding_lookup(variables['ent_re_embeddings'], t)
    e2_t = tf.nn.embedding_lookup(variables['ent_im_embeddings'], t)
    r1 = tf.nn.embedding_lookup(variables['rel_re_embeddings'], r)
    r2 = tf.nn.embedding_lookup(variables['rel_im_embeddings'], r)
    # Calculating score functions for all positive triples and negative triples
    res = tf.reduce_sum(_calc(e1_h, e2_h, e1_t, e2_t,
                              r1, r2), 1, keep_dims=False)
    loss_func = tf.reduce_mean(
        tf.nn.softplus(- y * res), 0, keep_dims=False)
    regul_func = tf.reduce_mean(e1_h ** 2) + tf.reduce_mean(e1_t ** 2) + tf.reduce_mean(
        e2_h ** 2) + tf.reduce_mean(e2_t ** 2) + tf.reduce_mean(r1 ** 2) + tf.reduce_mean(r2 ** 2)
    # Calculating loss to get what the framework will optimize
    return loss_func + options['lambda'] * regul_func


def predict(state, variables):
    predict_h = state.predict_h
    predict_t = state.predict_t
    predict_r = state.predict_r
    predict_h_e1 = tf.nn.embedding_lookup(
        variables['ent_re_embeddings'], predict_h)
    predict_t_e1 = tf.nn.embedding_lookup(
        variables['ent_re_embeddings'], predict_t)
    predict_r_e1 = tf.nn.embedding_lookup(
        variables['rel_re_embeddings'], predict_r)
    predict_h_e2 = tf.nn.embedding_lookup(
        variables['ent_im_embeddings'], predict_h)
    predict_t_e2 = tf.nn.embedding_lookup(
        variables['ent_im_embeddings'], predict_t)
    predict_r_e2 = tf.nn.embedding_lookup(
        variables['rel_im_embeddings'], predict_r)
    return -tf.reduce_sum(_calc(predict_h_e1, predict_h_e2, predict_t_e1, predict_t_e2, predict_r_e1, predict_r_e2), 1, keep_dims=True)
