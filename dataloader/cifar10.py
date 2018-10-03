import numpy as np
import pickle
import json
import math
from utils import helper

class DataLoader:
    
    def __init__(self):
        self.data, self.labels = self._load_training_data()
        self.test_data, self.test_labels = self._load_test_data()
        self.epoch = 0
        self.step = 0
        
    def num_step(self, batch_size: int) -> int:
        return math.floor(len(self.data)/batch_size)
        
    def next_batch(self, batch_size: int):
        data_size = len(self.data)
        next_step = self.step + 1
        if next_step * batch_size > data_size:
            self.epoch += 1
            self.step = 0
            next_step = 1
        batch_data = self.data[self.step*batch_size:next_step*batch_size]
        batch_labels = self.labels[self.step*batch_size:next_step*batch_size]
        self.step += 1
        return batch_data, batch_labels
        
    def _load_training_data(self):
        path = 'data/cifar-10-batches-py/'
        file_format = 'data_batch_{}'
        train_data = []
        train_labels = []
        with open(path + file_format.format(1), 'r') as f:
            for i in range(5):
                data, labels = self._get_data_from_file(path + file_format.format(i+1))
                train_data.extend(self._get_reshaped_data(data))
                train_labels.extend(labels)
        return np.array(train_data), np.array(train_labels)
    
    def _load_test_data(self):
        filepath = 'data/cifar-10-batches-py/test_batch'
        data, labels = self._get_data_from_file(filepath)
        data = self._get_reshaped_data(data)
        return data, np.array(labels)
    
    def _get_data_from_file(self, filepath):
        f = open(filepath, 'rb')
        raw_data = pickle.load(f, encoding='bytes')
        data, labels = self._get_data_and_labels(raw_data)
        return data, labels
    
    def _get_data_and_labels(self, dict_data):
        data = dict_data[b'data']
        labels = dict_data[b'labels']
        return data, labels
    
    def _get_reshaped_data(self, data):
        # 元データのshape check
        original_data_shape = [10000, 32*32*3]
        helper.assert_shape(data, original_data_shape)

        reshaped_data_shape = [10000, 32, 32, 3]
        reshaped_data = np.reshape(data, [10000, 3, 32, 32])
        reshaped_data = np.transpose(reshaped_data, [0, 2, 3, 1])

        # 修正したデータのshape check
        helper.assert_shape(reshaped_data, reshaped_data_shape)
        return reshaped_data
