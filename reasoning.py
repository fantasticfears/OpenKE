import config
import models
import tensorflow as tf
import numpy as np

con = config.Config()
con.set_test_flag(False)
con.set_in_path(b"./benchmarks/FB15K/")
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_nbatches(100)
con.init()
con.set_model(models.TransD)
r = config.TextReasonator(con, './res/model.vec.tf')

triplet = tuple('/m/0175yg	/music/genre/parent_genre	/m/0155w'.split('\t'))
print(triplet)
print(r.classify(triplet))
print(r.predict(triplet, direction='tail'))
