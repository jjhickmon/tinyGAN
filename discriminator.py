import numpy as np

class Discriminator:

	def __init__(self, learning_rate):
		self.learning_rate = learning_rate
		self.weights = np.array([np.random.normal(), np.random.normal(), np.random.normal(), np.random.normal()])
		self.bias = np.random.normal()

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# we want the real image prediction to be 1
	def error_real(self, prediction):
		return -np.log(prediction)
	
	def error_fake(self, prediction):
		return -np.log(1 - prediction)

	def forward(self, input):
		return self.sigmoid(np.dot(input, self.weights) + self.bias)

	def train_fake(self, prediction, fake):
		self.weights -= self.learning_rate * prediction * fake
		self.bias -= self.learning_rate * prediction

	def train_real(self, prediction, real):
		self.weights -= self.learning_rate * -(1 - prediction) * real
		self.bias -= self.learning_rate * -(1 - prediction)