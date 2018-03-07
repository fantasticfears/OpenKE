import sys
sys.path.append("../..")

import openke

if __name__ == '__main__':
  test = openke.EmbeddingsTest(openke.config.TestOptions('test.yml'))
  test.run()
