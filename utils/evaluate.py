from sklearn import metrics
import pandas as pd
import numpy as np

np.random.seed(13)


def evaluate(y_test, y_pred, PATH):
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


