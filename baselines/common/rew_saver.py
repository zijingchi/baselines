import numpy as np
import os
import pickle


class RewSaver(object):
    def __init__(self, path, max_len, start=0):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        self.max_len = max_len
        self.cur = start
        self.i = 0
        self.rew_buffer = []
        self.new_buffer = []

    def save(self):
        with open(os.path.join(self.path, 'rews_' + str(self.i) + '.pkl'), 'wb') as f:
            pickle.dump({'rews': self.rew_buffer, 'news': self.new_buffer}, f)

    def record(self, rews, news):
        n = len(rews)
        self.cur += n
        if self.cur > self.max_len:
            self.save()
            self.cur -= self.max_len
            self.rew_buffer = []
            self.new_buffer = []
            self.i += 1
        self.rew_buffer.extend(rews)
        self.new_buffer.extend(news)