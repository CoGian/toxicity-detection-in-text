import argparse
import tensorflow as tf
from transformers import *
import numpy as np
import gc
import os
import glob
from sklearn import metrics
import pandas as pd
from CostSensitiveHandling import stratification_undersample, rejection_sampling, example_weighting

parser = argparse.ArgumentParser()
parser.add_argument(
	"--data_path",
	"-d",
	help="path of the datasets",
	default='data/toy-roberta'
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

args = parser.parse_args()
data_path = args.data_path
MODEL = args.model_name
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)
MAX_LEN = int(args.max_len)
mode = args.mode
saving_path = args.save_path
BUFFER_SIZE = np.ceil(1804874 * 0.8)

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

"""# Load Datasets"""

TARGET_COLUMN = 'target'
TOXICITY_COLUMN = 'toxicity'

"""# Get datasets
"""


def get_dataset(PATH, mode=None, forTrain=False, forTest=False):
	filenames = glob.glob(PATH + '/*_input_ids.npy', recursive=False)
	for index, fname in enumerate(sorted(filenames)):
		if index == 0:
			input_ids = np.load(fname, allow_pickle=True, mmap_mode="r")
		else:
			input_ids = np.concatenate((input_ids, np.load(fname, allow_pickle=True, mmap_mode="r")), axis=0)

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
	gc.collect()
	if not forTest:

		filenames = glob.glob(PATH + '/*_sample_weights.npy', recursive=False)
		for index, fname in enumerate(sorted(filenames)):
			if index == 0:
				sample_weights = np.load(fname, allow_pickle=True)
			else:
				sample_weights = np.concatenate((sample_weights, np.load(fname, allow_pickle=True)), axis=0)

		if forTrain:
			if mode == "under_sampling":
				print("Under Sampling...")
				X = np.dstack((input_ids, attention_mask))
				X, labels = stratification_undersample(X, labels, per=0.66, dimensions=3)
				input_ids, attention_mask = np.dsplit(X, 2)
				input_ids = input_ids.reshape(1, -1)
				attention_mask = attention_mask.reshape(1, -1)
				print("New length of dataset", input_ids.shape[0])
			elif mode == "rejection_sampling":
				print("Rejection Sampling...")
				X = np.dstack((input_ids, attention_mask))
				X, labels = rejection_sampling(X, labels)
				input_ids, attention_mask = X.T
				print("New length of dataset", input_ids.shape[0])
			elif mode == "example_weighting":
				print("Weighting ..")
				sample_weights = example_weighting(labels)
			elif mode == "vanilla":
				pass

			global BUFFER_SIZE
			BUFFER_SIZE = len(input_ids)
		return tf.data.Dataset.from_tensor_slices((
			{"input_word_ids": input_ids, "input_mask": attention_mask},
			{"target": labels}, sample_weights))
	else:
		return tf.data.Dataset.from_tensor_slices({"input_word_ids": input_ids, "input_mask": attention_mask}).batch(
			BATCH_SIZE), labels

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
train_inputs_ds = get_dataset(PATH=os.path.join(data_path, 'train'), mode=mode, forTrain=True).repeat().shuffle(BUFFER_SIZE).batch(
	BATCH_SIZE).prefetch(AUTO)
gc.collect()
val_inputs_ds = get_dataset(PATH=os.path.join(data_path, 'val')).batch(BATCH_SIZE).cache().prefetch(AUTO)
gc.collect()
test_inputs_ds, y_test = get_dataset(PATH=os.path.join(data_path, 'test'), forTest=True)
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

model.fit(
	x=train_inputs_ds,
	validation_data=val_inputs_ds,
	epochs=EPOCHS,
	verbose=1,
	steps_per_epoch=n_steps)

y_pred = model.predict(test_inputs_ds, verbose=1)


def evaluate_csl(y_test, y_pred, PATH):
	y_pred = np.where(y_pred >= .5, 1, 0)

	with open(PATH + '/y_pred.npy', 'wb') as f:
		np.save(f, y_pred)

	cost_m = [[0.1, 2], [1, 0]]

	acc = metrics.accuracy_score(y_test, y_pred)
	print('Accuracy on test: {:f}'.format(acc))
	prec = metrics.precision_score(y_test, y_pred, average='macro')
	rec = metrics.recall_score(y_test, y_pred, average='macro')
	f1 = metrics.f1_score(y_test, y_pred, average='macro')
	confusion_matrix = metrics.confusion_matrix(y_test, y_pred).T
	loss = np.sum(confusion_matrix * cost_m)

	print(confusion_matrix)

	stats = {
		'Accuracy': acc,
		'Precision': prec,
		'Recall': rec,
		'F1': f1,
		'cost loss': loss}
	report_df = pd.DataFrame([stats])
	report_df = report_df.round(4)
	report_df.to_csv(PATH + '/report.csv')


evaluate_csl(y_test, y_pred, PATH=data_path)
