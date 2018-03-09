import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
from config.Config import Config
from heapq import *

class Reasonator(object):

    def __init__(self, config, tf_path):
        config.set_test_flag(False)
        config.set_import_files(tf_path)
        config.restore_tensorflow()
        self._config = config

    def _process_and_get_top(self, res, top):
        res = [(s[0], i) for i, s in enumerate(res)]
        heapify(res)
        return [i[1] for i in nsmallest(top, res, lambda x: x[0])]

    def predict(self, triplet, direction='head', top=10):
        if top < 1:
            raise

        h, r, t = triplet
        total_entity = self._config.lib.getEntityTotal()
        if direction == 'head':
            test_h = np.arange(total_entity, dtype=np.int64)
            test_t = np.repeat(t, total_entity)
        else:
            test_t = np.arange(total_entity, dtype=np.int64)
            test_h = np.repeat(h, total_entity)
        test_r = np.repeat(r, total_entity)
        res = self._config.test_step(test_h, test_t, test_r)
        res = self._process_and_get_top(res, top)
        return res

    def classify(self, triplet, tolerance=20):
        h, _, t = triplet
        return h in self.predict(triplet, direction='head', top=tolerance) and (
            t in self.predict(triplet, direction='tail', top=tolerance))

class TextReasonator(object):
    def __init__(self, config, tf_path):
        self.r = Reasonator(config, tf_path)
        with open(os.path.join(config.in_path, 'entity2id.txt')) as f:
            f.readline()
            self._entity_to_id = {}
            self._id_to_entity = {}
            for l in f.readlines():
                k,v = l.split('\t')
                self._entity_to_id[k] = int(v)
                self._id_to_entity[int(v)] = k
        with open(os.path.join(config.in_path, 'relation2id.txt')) as f:
            f.readline()
            self._relation_to_id = {}
            for l in f.readlines():
                k,v = l.split('\t')
                self._relation_to_id[k] = int(v)

    def _translate_triplet(self, triplet):
        h, r, t = triplet
        return (self._entity_to_id[h], self._relation_to_id[r], self._entity_to_id[t])

    def predict(self, triplet, direction='head', top=10):
        return [self._id_to_entity[x] for x in self.r.predict(self._translate_triplet(triplet), direction, top)]

    def classify(self, triplet, tolerance=20):
        return self.r.classify(self._translate_triplet(triplet), tolerance)
