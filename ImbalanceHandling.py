from random import choices, shuffle
import numpy
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KDTree
from sklearn.datasets import load_breast_cancer

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

# Usage
# sd = SMOTEDataset()
# my_X, my_y = ee.get_dataset(X, y)

class SMOTEDataset:
    def __init__(self):
        self.oversample = SMOTE()
    
    def get_dataset(self, X, y):
        X, y = self.oversample.fit_resample(X, y)
        return X, y

# Usage
# db = DensityBasedSamplingDataset()
# my_X, my_y = db.get_dataset(X, y, k)

class DensityBasedSamplingDataset:
    def __init__(self):
        pass

    def get_dataset(self, X, y, k):
        counter = Counter(y)
        if len(counter) != 2:
            print('Use the function with binary data.')
        
        max_class_id = list(counter.keys())[numpy.argmax(list(counter.values()))]
        min_class_id = list(counter.keys())[numpy.argmin(list(counter.values()))]
        min_size = min(counter.values())

        max_class = []
        min_class = []
        for i in range(len(X)):
            if y[i]==max_class_id:
                max_class.append(X[i])
            else:
                min_class.append(X[i])

        kdtree = KDTree(max_class)
        weights_raw = [self.get_sum_of_neighbors(kdtree, [x], k) for x in max_class]
        weight_sum = sum(weights_raw)
        weights = [w/weight_sum for w in weights_raw]
        sampled_max_class_ids = numpy.random.choice(list(range(len(max_class))), min_size, replace=False, p=weights)

        sampled_max_class = numpy.array(max_class)[sampled_max_class_ids]

        x_processed = min_class + sampled_max_class
        y_processed = [min_class_id]*min_size + [max_class_id]*min_size

        lz = list(zip(x_processed, y_processed))
        shuffle(lz)
        x_processed = numpy.array([elem[0] for elem in lz])
        y_processed = numpy.array([elem[1] for elem in lz])

        return x_processed, y_processed


    def get_sum_of_neighbors(self, kdtree, x, k):
        arr, _ = kdtree.query(x, k)
        return sum(arr.flatten())
