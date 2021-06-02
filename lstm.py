import argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from tensorflow.keras import utils
import numpy as np
import os
from sklearn import metrics
import pandas as pd
from ImbalanceHandling import *
from CostSensitiveHandling import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    "-d",
    help="path of the datasets",
    default='data/cleared_data'
)
parser.add_argument(
    "--batch_size",
    "-b",
    help="batch size",
    default=256
)
parser.add_argument(
    "--epochs",
    "-e",
    help="epochs",
    default=2
)
parser.add_argument(
    "--max_len",
    "-ml",
    help="max length of sequence",
    default=128
)
parser.add_argument(
    "--mode",
    "-md",
    help="mode of cost-sensitivity learning or class imbalance",
    default="vanilla"
)
parser.add_argument(
    "--save_path",
    "-s",
    help="path to save results",
    default="vanilla_results"
)
#args = parser.parse_args()
data_path = 'data/cleared_data'
BATCH_SIZE = 256
EPOCHS = 10
MAX_LEN = 128
saving_path = "vanilla_results"
BUFFER_SIZE = np.ceil(1804874 * 0.8)
N_VOTERS = 3

seed = 13
tf.random.set_seed(seed)
np.random.seed(seed)

if not os.path.exists(saving_path):
    os.mkdir(saving_path)
print(tf.__version__)
"""  # TPU Configs"""

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

def build_lstm_model():
    # create and fit the LSTM network
    model = Sequential()
    model.add(Embedding(100000, 300, input_length=MAX_LEN, trainable=True))
    model.add(LSTM(100, input_shape=(300, MAX_LEN), return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def custom_loss(y_true, y_pred):
  pred_idx = tf.argmax(y_pred, axis=1, output_type=tf.int32)
  indices = tf.stack([tf.reshape(pred_idx, (-1)),
                      tf.reshape(tf.cast(y_true, tf.int32), (-1,))
                      ], axis=1)
  batch_weights = tf.gather_nd(weights, indices)

  return batch_weights*tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def print_metrics(y_true, y_pred):
    y_true = [np.argmax(out) for out in y_true]
    y_pred = [np.argmax(out) for out in y_pred]
    print("F1 macro: ", end='')
    print(metrics.f1_score(y_true, y_pred, average='macro'))
    print("Accuracy: ", end='')
    print(metrics.accuracy_score(y_true, y_pred))

data_train = pd.read_csv(data_path+'/train_cleared.csv')
x_train, y_train = data_train['comment_text'].apply(str).to_list(), data_train['target'].to_numpy()

data_test = pd.read_csv(data_path+'/test_cleared.csv')
x_test, y_test = data_test['comment_text'].apply(str).to_list(), data_test['toxicity'].to_numpy()

mode = 3

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100000, lower=False)
tokenizer.fit_on_texts(x_train+x_test)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

y_train = np.where(y_train>=0.5, 1, 0)
y_test = np.where(y_test>=0.5, 1, 0)

if mode==0:
    x_train, y_train = stratification_undersample(x_train, y_train)
elif mode==1:
    x_train, y_train = rejection_sampling(x_train, y_train)
elif mode==2:
    weights = example_weighting(y_train)
elif mode==3:
    ee = EasyEnsembleDataset(N_VOTERS)
    datasets = ee.get_dataset(x_train, y_train)
elif mode==4:
    over = RandomOversampledDataset()
    x_train, y_train = over.get_dataset(x_train, y_train)
elif mode==5:
    under = RandomUndersampledDataset()
    x_train, y_train = under.get_dataset(x_train, y_train)


if mode != 3:
    with strategy.scope():
      model = build_lstm_model()
    print(model.summary())
    x_train, x_test = np.array(x_train), np.array(x_test)

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)
    if mode != 2:
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=5000, verbose=1)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=5000, verbose=1, sample_weight=weights)

    print("Training metrics")
    y_pred_train = model.predict(x_train)
    print_metrics(y_train, y_pred_train)

    print("Testing metrics")
    y_pred = model.predict(x_test)
    print_metrics(y_test, y_pred)
elif mode == 3:
    output_train = []
    output_test = []
    for voter in range(N_VOTERS):
        with strategy.scope():
            model = build_lstm_model()
        #print(model.summary())
        x_train, y_train = datasets[voter]

        y_train = utils.to_categorical(y_train, 2)
        y_test = utils.to_categorical(y_test, 2)  

        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=5000, verbose=1)

        y_pred_train = model.predict(x_train)
        y_pred = model.predict(x_test)
        
        output_train.append([np.argmax(y) for y in y_pred_train])
        output_test.append([np.argmax(y) for y in y_pred])

    output_train = np.array(output_train)
    print(output_train.shape)
    output_test = np.array(output_test)
    output_train = output_train.reshape(output_train.shape[0], output_train.shape[1]).T
    output_test = output_test.reshape(output_test.shape[0], output_test.shape[1]).T

    majorities_train = [np.argmax(np.bincount(column)) for column in output_train]
    majorities_test = [np.argmax(np.bincount(column)) for column in output_test]

    print("Testing metrics")
    y_pred = model.predict(x_test)
    print_metrics(y_test, majorities_test)
