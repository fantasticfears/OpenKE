import config
import models
import json
import os 
import sys

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path(f"/data/wikidata/{sys.argv[1]}/")
#True: Input test files from the same folder.
# con.set_log_on(1)
con.set_work_threads(16)
con.set_train_times(100)
con.set_nbatches(1000)	
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_margin(1.0)
con.set_save_steps(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_valid_steps(10000)
con.set_opt_method("SGD")
#Model parameters will be exported via torch.save() automatically.
con.set_checkpoint_dir("./res/transe.pt")
#Model parameters will be exported to json files automatically.
con.set_result_dir("./res/embedding.vec.json")
con.init()
con.set_test_model(models.TransE)
# con.trainModel = nn.DataParallel(con.trainModel)
con.test()