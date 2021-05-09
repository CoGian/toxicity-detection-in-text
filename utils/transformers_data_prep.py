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

train_df_chunk = pd.read_csv(os.path.join(data_path, "train_cleared.csv"), usecols=[TARGET_COLUMN, 'comment_text'], chunksize=40000)
#train_df = train_df[:1000]
test_public_df_chunk = pd.read_csv(os.path.join(data_path, "test_public_cleared.csv"), usecols=['comment_text'], chunksize=40000)
# test_public_df = test_public_df.loc[:, ['toxicity', 'comment_text']].dropna()[:500]
test_private_df_chunk = pd.read_csv(os.path.join(data_path, "test_private_cleared.csv"), usecols=['comment_text'], chunksize=40000)
# test_private_df = test_private_df.loc[:, ['toxicity', 'comment_text']].dropna()[:500]
tokenizer_transformer = AutoTokenizer.from_pretrained(model_name)


def encode_examples(df, PATH, index, sample_weights=None, labels=None, forTest=False):
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
	with open(PATH + '/' + str(index) + '_input_ids.npy', 'wb') as filehandle:
		# store the data as binary data stream
		np.save(filehandle, np.asarray(_input['input_ids']))

	with open(PATH + '/' + str(index) + '_input_mask.npy', 'wb') as filehandle:
		# store the data as binary data stream
		np.save(filehandle, np.asarray(_input['attention_mask']))

	del _input
	gc.collect()
	if not forTest:
		with open(PATH + '/' + str(index) + '_labels.npy', 'wb') as filehandle:
			# store the data as binary data stream
			np.save(filehandle, np.asarray(labels))
		with open(PATH + '/' + str(index) + '_sample_weights.npy', 'wb') as filehandle:
			# store the data as binary data stream
			np.save(filehandle, np.asarray(sample_weights))


print("Start preprocess..")
for index, chunk in enumerate(train_df_chunk):
	y_train = chunk[TARGET_COLUMN].values.reshape((-1, 1))
	sample_weights_train = np.ones(len(chunk), dtype=np.float32)

	encode_examples(
		df=chunk,
		PATH=os.path.join(saving_path, 'train'),
		index=index,
		sample_weights=sample_weights_train,
		labels=y_train,
		)

print("Finished train data..")

for index, chunk in enumerate(test_private_df_chunk):

	encode_examples(
		df=chunk,
		PATH=os.path.join(saving_path, 'test_private'),
		index=index,
		forTest=True,
		)

print("Finished test private data..")

for index, chunk in enumerate(test_public_df_chunk):

	encode_examples(
		df=chunk,
		PATH=os.path.join(saving_path, 'test_public'),
		index=index,
		forTest=True,
		)
print("Finished test public data..")