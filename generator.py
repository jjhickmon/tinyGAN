import numpy as np

class Generator:

	def __init__(self, learning_rate):
		self.learning_rate = learning_rate
		self.weights = np.array([np.random.normal(), np.random.normal(), np.random.normal(), np.random.normal()])
		self.biases = np.array([np.random.normal(), np.random.normal(), np.random.normal(), np.random.normal()])

	# The sigmoid activation function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# the generator wants the descriminator to thing a fake image is real
	def error(self, prediction_fake):
		return -np.log(prediction_fake)

	def forward(self, z):
		return self.sigmoid(z * self.weights + self.biases)

	def train(self, z, prediction, fake, discriminator_weights):
		factor = -(1 - prediction) * discriminator_weights * fake * (1 - fake)
		self.weights -= self.learning_rate * factor * z
		self.biases -= self.learning_rate * factor
