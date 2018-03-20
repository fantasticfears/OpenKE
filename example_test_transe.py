import config
import models
import tensorflow as tf
import json
import openke
import os.path

# (1) Set import files and OpenKE will automatically load models via tf.Saver().

def test_step(sess, predict_op, feed_dict):
    return sess.run(predict_op, feed_dict)

helper = openke.TestHelper(os.path.join('./benchmarks/FB15K/'))

with tf.Session().as_default() as sess:
    predict = openke.test_model_fn(helper, 'ComplEx', {'entity_size': 14951, 'relation_size': 1345, 'embedding_size': 100})
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('./res', 'model.ckpt'))

    total = helper.total_test_case
    for times in range(total):
        helper.prepare_batch(direction='head')
        res = test_step(sess, predict, helper.feed_dict)
        helper.test_batch(res, direction='head')

        helper.prepare_batch(direction='tail')
        res = test_step(sess, predict, helper.feed_dict)
        helper.test_batch(res, direction='tail')

        print(times)
    helper.test_stat()

# (2) Read model parameters from json files and manually load parameters.
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# f = open("./res/embedding.vec.json", "r")
# content = json.loads(f.read())
# f.close()
# con.set_parameters(content)
# con.test()

# (3) Manually load models via tf.Saver().
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# con.import_variables("./res/model.vec.tf")
# con.test()
