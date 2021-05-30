from random import choices, shuffle
import numpy
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KDTree
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

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
        self.oversample = SMOTE(n_jobs=-1)
    
    def get_dataset(self, x_train, y_train, x_test, y_test):
        tfidf = TfidfVectorizer(min_df=5)
        matrix = numpy.array(tfidf.fit_transform(x_train).todense())
        x_transformed = []
        feature_names = tfidf.get_feature_names()
        for i in range(len(x_train)):
            sentence = x_train[i]
            word_list = word_tokenize(sentence)
            word_list_transformed = []
            for w in word_list:
                if w in feature_names:
                    pos = feature_names.index(w)
                    word_list_transformed.append(matrix[i][pos])
                else:
                    pos = -1
                    word_list_transformed.append(0.0)
            x_transformed.append(word_list_transformed)
        x_transformed = tf.keras.preprocessing.sequence.pad_sequences(x_transformed, maxlen=MAX_LEN)
        x_train, y_train = self.oversample.fit_resample(numpy.array(x_transformed), y_train)

        matrix_test = numpy.array(tfidf.transform(x_test).todense())
        x_transformed_test = []
        for i in range(len(x_test)):
            sentence = x_test[i]
            word_list = word_tokenize(sentence)
            word_list_transformed = []
            for w in word_list:
                if w in feature_names:
                    pos = feature_names.index(w)
                    word_list_transformed.append(matrix_test[i][pos])
                else:
                    pos = -1
                    word_list_transformed.append(0.0)
            x_transformed_test.append(word_list_transformed)
        x_transformed_test = tf.keras.preprocessing.sequence.pad_sequences(x_transformed_test, maxlen=MAX_LEN)
        x_test, y_test = self.oversample.fit_resample(numpy.array(x_transformed_test), y_test)

        return x_train, y_train, x_test, y_test

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

        kdtree = KDTree(numpy.array(max_class))
        weights_raw = [self.get_sum_of_neighbors(kdtree, [x], k) for x in max_class]
        weight_sum = sum(weights_raw)
        weights = [w/weight_sum for w in weights_raw]
        sampled_max_class_ids = numpy.random.choice(list(range(len(max_class))), min_size, replace=False, p=weights)

        sampled_max_class = numpy.array(max_class)[sampled_max_class_ids]
        x_processed = numpy.concatenate((numpy.array(min_class), sampled_max_class))
        y_processed = numpy.array([min_class_id]*min_size + [max_class_id]*min_size)

        return x_processed, y_processed

    def get_sum_of_neighbors(self, kdtree, x, k):
        arr, _ = kdtree.query(x, k)
        return sum(arr.flatten())
