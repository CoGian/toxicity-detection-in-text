import argparse
import numpy as np
import gc
import os
import pandas as pd
from CostSensitiveHandling import stratification_undersample, rejection_sampling, example_weighting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from utils.evaluate import evaluate
parser = argparse.ArgumentParser()

parser.add_argument(
	"--data_path",
	"-d",
	help="path of the datasets",
	default='data'
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
EPOCHS = int(args.epochs)
MAX_LEN = int(args.max_len)
mode = args.mode
saving_path = args.save_path

if not os.path.exists(saving_path):
	os.mkdir(saving_path)

train = pd.read_csv(os.path.join(data_path, "train_cleared.csv"))
test = pd.read_csv(os.path.join(data_path, "test_cleared.csv"))

x_test = test["comment_text"].astype(str).values.reshape(-1, 1)
y_test = np.where(test["toxicity"].values.reshape((-1, 1)) >= .5, 1, 0)

del test
gc.collect()
if mode == "under_sampling":
	print("Under Sampling...")
	X, labels = stratification_undersample(
		train["comment_text"].astype(str).values.reshape(-1, 1),
		np.where(train["target"].values.reshape((-1, 1)) >= .5, 1, 0),
		per=0.66,
		dimensions=2)

	class_weights = {0: 1, 1: 1}
	print("New length of dataset", X.shape[0])
elif mode == "rejection_sampling":
	print("Rejection Sampling...")
	X, labels = rejection_sampling(
		train["comment_text"].astype(str).values.reshape(-1, 1),
		np.where(train["target"].values.reshape((-1, 1)) >= .5, 1, 0))

	class_weights = {0: 1, 1: 1}
	print("New length of dataset", X.shape[0])
elif mode == "example_weighting":
	print("Weighting ..")
	X = train["comment_text"].astype(str).values.reshape(-1, 1)
	labels = np.where(train["target"].values.reshape((-1, 1)) >= .5, 1, 0)

	class_weights = {0: 1, 1: 2}
elif mode == "vanilla":
	X = train["comment_text"].astype(str).values.reshape(-1, 1)
	labels = np.where(train["target"].values.reshape((-1, 1)) >= .5, 1, 0)

	class_weights = {0: 1, 1: 1}

del train
gc.collect()
clf = make_pipeline(TfidfVectorizer(), LinearSVC(
	random_state=0,
	tol=1e-5,
	verbose=1,
	class_weight=class_weights))

clf.fit(X.ravel(), labels)
y_pred = clf.predict(x_test.ravel())
evaluate(y_test, y_pred, saving_path)
