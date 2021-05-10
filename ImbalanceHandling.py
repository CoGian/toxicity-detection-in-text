from random import choices, shuffle
import numpy
from collections import Counter

class EasyEnsembleDataset:
    def __init__(self, n_datasets):
        self.n_datasets = n_datasets
    
    def get_datasets(self, X, y):
        result = []
        for iter in range(self.n_datasets):
            counter = Counter(y)
            if len(counter) != 2:
                print('Fit with binary data. Fitting failed...')
            
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
            
            x_train = min_class + max_class
            y_train = [min_class_id]*min_size + [max_class_id]*min_size

            lz = list(zip(x_train, y_train))
            shuffle(lz)
            x_train = numpy.array([elem[0] for elem in lz])
            y_train = numpy.array([elem[1] for elem in lz])

            result.append((x_train, y_train))
        
        return result

# Usage
# ee = EasyEnsembleDataset(n_datasets)
# datasets = ee.get_datasets(X, y)
# returns n_datasets pairs of X, y


