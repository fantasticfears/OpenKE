import sys
sys.path.append("../..")

import openke

if __name__ == '__main__':
  train = openke.TrainFlow(openke.config.TrainOptions('train.yml'))
  train.train()
