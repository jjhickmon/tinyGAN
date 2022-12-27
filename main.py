from generator import Generator
from discriminator import Discriminator
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt

def visualize(grid):
	color1 = (grid[0] * 255, grid[0] * 255, grid[0] * 255)
	color2 = (grid[1] * 255, grid[1] * 255, grid[1] * 255)
	color3 = (grid[2] * 255, grid[2] * 255, grid[2] * 255)
	color4 = (grid[3] * 255, grid[3] * 255, grid[3] * 255)
	pygame.draw.rect(screen, color1, (0, 0, 200, 200))
	pygame.draw.rect(screen, color2, (200, 0, 200, 200))
	pygame.draw.rect(screen, color3, (0, 200, 200, 200))
	pygame.draw.rect(screen, color4, (200, 200, 200, 200))

pygame.init()
screen = pygame.display.set_mode((400, 400))

# np.random.seed(42)
learning_rate = 0.01
epochs = 1000
generator = Generator(learning_rate)
discriminator = Discriminator(learning_rate)
gen_error = []
discr_error_fake = []
discr_error_real = []
acc_real = []
acc_fake = []
training = [np.array([1,0,0,1]),
        	np.array([0.9,0.1,0.2,0.8]),
        	np.array([0.9,0.2,0.1,0.8]),
        	np.array([0.8,0.1,0.2,0.9]),
        	np.array([0.8,0.2,0.1,0.9])]

for epoch in range(epochs):
	# train
	for real in training:
		z = np.random.rand()
		fake = generator.forward(z)
		
		prediction = discriminator.forward(fake)
		discriminator.train_fake(prediction, fake)
		generator.train(z, prediction, fake, discriminator.weights)

		prediction = discriminator.forward(real)
		discriminator.train_real(prediction, real)
	
	# log data
	fake = generator.forward(z)
	real = training[np.random.randint(0, 5)]
	prediction_fake = discriminator.forward(fake)
	prediction_real = discriminator.forward(real)
	gen_error.append(generator.error(prediction_fake))
	discr_error_fake.append(discriminator.error_fake(prediction_fake))
	discr_error_real.append(discriminator.error_real(prediction_real))
	acc_real.append(prediction_real)
	acc_fake.append(1 - prediction_fake)

test = generator.forward(random.random())

# visualize results
visualize(test)
pygame.display.flip()

# plot error functions
figure, (err, acc) = plt.subplots(1, 2)
err.plot(gen_error, label="gen", color="green")
err.plot(discr_error_fake, label="d-fake", color="red")
err.plot(discr_error_real, label="d-real", color="blue")
err.set_title("Error Functions")
err.set_xlabel("epoch")
err.set_ylabel("error")
err.legend()

acc.plot(acc_fake, label="acc-fake", color="red")
acc.plot(acc_real, label="acc-real", color="green")
acc.set_title("Discriminator Accuracy")
acc.set_xlabel("epoch")
acc.set_ylabel("accuracy")
acc.legend()
plt.show()
