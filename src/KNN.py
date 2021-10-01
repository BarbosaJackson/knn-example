import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class KNN(object):
	def __init__(self, dataset, head):
		self.dataset = dataset
		self.head = head
		self.x = None
		self.y = None
		self.iris_classifier = None
		self.best_k = 0
		self.accuracy = 0

	def load_data_set(self):
		df = pd.read_csv(self.dataset, names = self.head)
		self.x = df[df.columns.difference(['Class'])].values
		self.y = df['Class'].values

	def train(self, k):
		self.iris_classifier = KNeighborsClassifier(n_neighbors = k, p=3)
		self.iris_classifier.fit(self.x, self.y)

	def get_accuracy(self, k):
		scores_dt = cross_val_score(self.iris_classifier, self.x, self.y, scoring='accuracy', cv=5)
		return scores_dt.mean()

	def calibration(self, max_k):
		print("k    | accuracy")
		print("-----------------")

		cur_k = 1
		while(cur_k <= max_k):
			self.train(cur_k)
			cur_accuracy = self.get_accuracy(cur_k)
			if(cur_accuracy > self.accuracy):
				self.accuracy = cur_accuracy
				self.best_k = cur_k
			cur_k += 1

			table_str = str(cur_k - 1)
			if(cur_k - 1 > 9):
				table_str += "   |"
			else:
				table_str += "    |"
			print("{0} {1:.6f}".format(table_str, cur_accuracy))
		print('best k: {0} with {1:.6f} accuracy'.format(self.best_k, self.accuracy))


		self.train(self.best_k)

	def predict_data(self, data):
		result = self.iris_classifier.predict(data)
		return result