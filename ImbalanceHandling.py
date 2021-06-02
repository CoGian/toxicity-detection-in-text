from random import choices, shuffle
import random
import numpy
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import time

# Usage
# ee = EasyEnsembleDataset(n_datasets)
# datasets = ee.get_dataset(X, y)
# returns n_datasets pairs of X, y

class EasyEnsembleDataset:
    def __init__(self, n_datasets):
        self.n_datasets = n_datasets
    
    def get_dataset(self, X, y):
        result = []
        for _ in range(self.n_datasets):
            random.seed(time.time())
            counter = Counter(y)
            if len(counter) != 2:
                print('Use the function with binary data.')
            
            min_size = min(counter.values())
            max_class_id = list(counter.keys())[numpy.argmax(list(counter.values()))]
            min_class_id = list(counter.keys())[numpy.argmin(list(counter.values()))]

            max_class = []
            min_class = []
            for i in range(len(X)):
                if y[i] == max_class_id:
                    max_class.append(X[i])
                else:
                    min_class.append(X[i])
            max_class = choices(max_class, k=min_size)
            
            x_processed = min_class + max_class
            y_processed = [min_class_id]*min_size + [max_class_id]*min_size

            lz = list(zip(x_processed, y_processed))
            shuffle(lz)
            x_processed = numpy.array([elem[0] for elem in lz])
            y_processed = numpy.array([elem[1] for elem in lz])

            result.append((x_processed, y_processed))
        
        return result

class RandomUndersampledDataset:
    def __init__(self):
        self.undersampler = RandomUnderSampler()

    def get_dataset(self, X, y):
        X, y = self.undersampler.fit_resample(X, y)
        return X, y

class RandomOversampledDataset:
    def __init__(self):
        self.oversampler = RandomOverSampler()

    def get_dataset(self, X, y):
        X, y = self.oversampler.fit_resample(X, y)
        return X, y