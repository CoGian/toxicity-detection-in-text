import argparse

import tensorflow as tf
from transformers import *
import numpy as np
import gc
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
	"--data_path",
	"-d",
	help="path of the datasets",
	default='data'
)
parser.add_argument(
	"--checkpoint_dir",
	"-cd",
	help="path to save model checkpoints",
	default='./roberta-base'
)
parser.add_argument(
	"--model_name",
	"-m",
	help="name of transformer model",
	default='roberta-base'
)
parser.add_argument(
	"--batch_size",
	"-b",
	help="batch size",
	default=128
)
parser.add_argument(
	"--epochs",
	"-e",
	help="epochs",
	default=10
)

parser.add_argument(
	"--max_len",
	"-ml",
	help="max length of sequence",
	default=128
)

args = parser.parse_args()
data_path = args.data_path
MODEL = args.model_name
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MAX_LEN = args.max_len
checkpoint_dir = args.checkpoint_dir
BUFFER_SIZE = np.ceil(1804874 * 0.8)

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

"""# Load Datasets"""

TARGET_COLUMN = 'target'
TOXICITY_COLUMN = 'toxicity'

"""# Get datasets
"""


def get_dataset(PATH):
	filenames = glob.glob(PATH + '/*_input_ids.npy', recursive=False)
	for index, fname in enumerate(sorted(filenames)):
		if index == 0:
			input_ids = np.load(fname, allow_pickle=True)
		else:
			input_ids = np.concatenate((input_ids, np.load(fname, allow_pickle=True)), axis=0)

	filenames = glob.glob(PATH + '/*_input_mask.npy', recursive=False)
	for index, fname in enumerate(sorted(filenames)):
		if index == 0:
			attention_mask = np.load(fname, allow_pickle=True)
		else:
			attention_mask = np.concatenate((attention_mask, np.load(fname, allow_pickle=True)), axis=0)

	filenames = glob.glob(PATH + '/*_labels.npy', recursive=False)
	for index, fname in enumerate(sorted(filenames)):
		if index == 0:
			labels = np.load(fname, allow_pickle=True)
		else:
			labels = np.concatenate((labels, np.load(fname, allow_pickle=True)), axis=0)

	filenames = glob.glob(PATH + '/*_sample_weights.npy', recursive=False)
	for index, fname in enumerate(sorted(filenames)):
		if index == 0:
			sample_weights = np.load(fname, allow_pickle=True)
		else:
			sample_weights = np.concatenate((sample_weights, np.load(fname, allow_pickle=True)), axis=0)

	return tf.data.Dataset.from_tensor_slices((
		{"input_word_ids": input_ids, "input_mask": attention_mask},
		{"target": labels}, sample_weights))


"""# RoBERTa-with-max-avg-pool
## Create Model
"""


def createTLmodel(transformer_layer):
	input_word_ids = tf.keras.layers.Input(
		shape=(MAX_LEN,),
		dtype=tf.int32,
		name="input_word_ids")
	input_mask = tf.keras.layers.Input(
		shape=(MAX_LEN,),
		dtype=tf.int32,
		name="input_mask")

	outputs = transformer_layer([input_word_ids, input_mask])
	avg_pool = tf.keras.layers.GlobalAveragePooling1D()(outputs.last_hidden_state)
	x = tf.keras.layers.Dropout(0.1)(avg_pool)
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	result = tf.keras.layers.Dense(1, activation='sigmoid', name='target')(x)

	model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=[result])
	model.compile(
		loss=tf.keras.losses.BinaryCrossentropy(),
		optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
		metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	return model


"""## RoBERTa - base"""

AUTO = tf.data.experimental.AUTOTUNE
train_inputs_ds = get_dataset(os.path.join(data_path, MODEL + '/train')).repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
gc.collect()
val_inputs_ds = get_dataset(os.path.join(data_path, MODEL + '/val')).batch(BATCH_SIZE).cache().prefetch(AUTO)
gc.collect()

with strategy.scope():
	transformer_layer = TFAutoModel.from_pretrained(MODEL)
	model = createTLmodel(transformer_layer)
model.summary()
tf.keras.utils.plot_model(
	model,
	show_shapes=True,
	show_layer_names=False,
	to_file=MODEL + '.png')

n_steps = BUFFER_SIZE // BATCH_SIZE
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_prefix,
	save_weights_only=False,
	monitor='val_loss',
	mode='max',
	save_best_only=True)


model.fit(
	x=train_inputs_ds,
	validation_data=val_inputs_ds,
	epochs=EPOCHS,
	verbose=1,
	steps_per_epoch=n_steps,
	callbacks=[model_checkpoint_callback])
