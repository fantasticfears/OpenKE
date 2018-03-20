import config
import models
import tensorflow as tf
import numpy as np

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.
con.set_test_flag(True)

con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
#Model parameters will be exported to json files automatically.
#Initialize experimental settings.
con.init()
con.set_model(models.TransD)

con.set_import_files('./res/model.vec.tf')
con.restore_tensorflow()
#Set the knowledge embedding model
#Train the model.
#To test models after training needs "set_test_flag(True)".
con.test()

