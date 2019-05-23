import config
import models
import json
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K237/")
#True: Input test files from the same folder.
con.set_log_on(1)
con.set_work_threads(16)
con.set_train_times(200)
con.set_nbatches(28)	
con.set_test_link_prediction(True)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_margin(1.0)
con.set_ent_neg_rate(14)
con.set_rel_neg_rate(6)
con.set_opt_method("SGD")
#Model parameters will be exported via torch.save() automatically.
con.set_import_files("./res/complex.pt")
#Model parameters will be exported to json files automatically.
con.init()
con.set_model(models.ComplEx)
con.test()
