import numpy as np
import pickle


class Opportunity():
	def _read_data(self):
		with open('opp_data.pkl', 'r') as fp:
			dataset = pickle.load(fp)
		return dataset

	def __init__(self):
		dataset = self._read_data()

		self.train_input = dataset['train_input']
		self.train_label = dataset['train_label']
		self.test_input = dataset['test_input']
		self.test_label = dataset['test_label']

	def train_next_batch(self, batch_size):
		shuffle_index = np.random.permutation(len(self.train_input))

		train_input_batch = self.train_input[shuffle_index[:batch_size]]
		train_label_batch = np.reshape(self.train_label[shuffle_index[:batch_size]], [batch_size])

		return train_input_batch, train_label_batch

	def test_data(self):
		return self.test_input, self.test_label


def main():
	opp = Opportunity()

	xs, ys = opp.train_next_batch(50)
	print(np.shape(xs))
	print(np.shape(ys))

	xs, ys = opp.test_data()
	print(np.shape(xs))
	print(np.shape(ys))

if __name__ == '__main__':
	main()
