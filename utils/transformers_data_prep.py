import gc

import pandas as pd
from transformers import *
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
	"--data_path",
	"-d",
	help="path of the datasets",
	default='data'
)
parser.add_argument(
	"--model_name",
	"-m",
	help="name of transformer model",
	default='roberta-base'
)
parser.add_argument(
	"--max_len",
	"-ml",
	help="max length of sequence",
	default=128
)
args = parser.parse_args()
data_path = args.data_path
model_name = args.model_name
MAX_LEN = args.max_len

saving_path = os.path.join(data_path, model_name)

if not os.path.exists(saving_path):
	os.mkdir(saving_path)

"""# Load Datasets"""
TARGET_COLUMN = 'target'
TOXICITY_COLUMN = 'toxicity'

train_df = pd.read_csv(os.path.join(data_path, "train_cleared.csv"))
# train_df = train_df[:1000]
test_public_df = pd.read_csv(os.path.join(data_path, "test_public_cleared.csv"))
# test_public_df = test_public_df.loc[:, ['toxicity', 'comment_text']].dropna()[:500]
test_private_df = pd.read_csv(os.path.join(data_path, "test_private_cleared.csv"))
# test_private_df = test_private_df.loc[:, ['toxicity', 'comment_text']].dropna()[:500]

# split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=13, shuffle=True)

y_train = train_df[TARGET_COLUMN].values.reshape((-1, 1))
y_val = val_df[TARGET_COLUMN].values.reshape((-1, 1))
y_public_test = test_public_df[TOXICITY_COLUMN].values.reshape((-1, 1))
y_private_test = test_private_df[TOXICITY_COLUMN].values.reshape((-1, 1))

train_df[TARGET_COLUMN] = np.where(train_df[TARGET_COLUMN] >= 0.5, 1, 0)
test_public_df[TOXICITY_COLUMN] = np.where(test_public_df[TOXICITY_COLUMN] >= 0.5, 1, 0)
test_private_df[TOXICITY_COLUMN] = np.where(test_private_df[TOXICITY_COLUMN] >= 0.5, 1, 0)
val_df[TARGET_COLUMN] = np.where(val_df[TARGET_COLUMN] >= 0.5, 1, 0)

gc.collect()
"""# Get datasets"""

tokenizer_transformer = AutoTokenizer.from_pretrained(model_name)

sample_weights_train = np.ones(len(train_df), dtype=np.float32)
sample_weights_val = np.ones(len(val_df), dtype=np.float32)


def encode_examples(df, PATH, sample_weights=None, labels=None, forTest=False):
	# prepare list, so that we can build up final TensorFlow dataset from slices.

	if not os.path.exists(PATH):
		os.mkdir(PATH)

	_input = tokenizer_transformer.batch_encode_plus(
		df['comment_text'].astype(str).values.tolist(),
		add_special_tokens=True,  # add [CLS], [SEP]
		max_length=MAX_LEN,  # max length of the text
		padding='max_length',  # add [PAD] tokens
		truncation=True,
		return_attention_mask=True  # add attention mask to not focus on pad tokens
		)
	del df
	gc.collect()
	with open(PATH + '/input_ids.npy', 'wb') as filehandle:
		# store the data as binary data stream
		np.save(filehandle, np.asarray(_input['input_ids']))

	with open(PATH + '/input_mask.npy', 'wb') as filehandle:
		# store the data as binary data stream
		np.save(filehandle, np.asarray(_input['attention_mask']))

	del _input
	gc.collect()
	if not forTest:
		with open(PATH + '/labels.npy', 'wb') as filehandle:
			# store the data as binary data stream
			np.save(filehandle, np.asarray(labels))
		with open(PATH + '/sample_weights.npy', 'wb') as filehandle:
			# store the data as binary data stream
			np.save(filehandle, np.asarray(sample_weights))


print("Start preprocess..")
encode_examples(train_df,
				os.path.join(saving_path, 'train'),
				sample_weights_train, y_train)
print("Finished train data..")
encode_examples(val_df,
				os.path.join(saving_path, 'val'),
				sample_weights_val, y_val)
print("Finished val data..")
encode_examples(test_private_df,
				os.path.join(saving_path, 'test_private'),
				forTest=True)
print("Finished test private data..")
encode_examples(test_public_df,
				os.path.join(saving_path, 'test_public'),
				forTest=True)
print("Finished test public data..")