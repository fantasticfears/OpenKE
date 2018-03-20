import openke
import models
import tensorflow as tf
import os.path
import time

def train_step(sess, train_op, loss_op, summary_op, feed_dict):
    return sess.run(
        [summary_op, loss_op, train_op], feed_dict)


def valid_step(sess, valid_op, feed_dict):
    return sess.run(
        [valid_op], feed_dict)


train_flow = openke.TrainFlow('train.yml')
for step_config in train_flow.config.flow:
    train_flow.reset(step_config)
    loss_ops, train_op, summary_op, variables = openke.train_model_gpu_fn(
        step_config, train_flow)

    print(variables)
    step = 0
    feed_dict = train_flow.train_feed_dict
    with tf.Session().as_default() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        summary_writer = tf.summary.FileWriter(step_config.export_path, sess.graph)

        for times in range(step_config.train_iterations):
            res = 0.0
            start_time = time.time()
            for batch in range(step_config.nbatches):
                train_flow.sampling(step_config)
                step += 1
                summary_str, loss, _ = train_step(
                    sess, train_op, loss_ops, summary_op, feed_dict)
                res += sum(loss)
                summary_writer.add_summary(summary_str, step)

            print("Iterations: " + str(times))
            print("Loss: " + str(res))
            print("Time: " + str(time.time() - start_time))
            # if self.exportName != None and (self.export_steps != 0 and times % self.export_steps == 0):
            #     self.save_tensorflow()
            # print(valid_step())

        if step_config.export_path:
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(step_config.export_path, 'model.ckpt'))
        if step_config.export_json is True:
            openke.save_variables_to_json(sess, variables, step_config.export_path)
