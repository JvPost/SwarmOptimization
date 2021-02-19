from ast import parse
from classes import Particle, Population, Clustering, fitness
import numpy as np
# import matplotlib.pyplot as plt # for some reason says can't find the module on my pc, idk why.

# helper functions
def createArtificialProblem1():
    return np.array([[i[0], i[1]] for i in zip(np.random.uniform(-1, 1, 400), np.random.uniform(-1, 1, 400))])
    
def labelFunc1(x):
    x1 = x[0]
    x2 = x[1]
    return int((x1 >= 0.7) or ((x1<=0.3) and (x2 >= (-0.2 - x1))))

# problem data
X = createArtificialProblem1()
y = np.array([labelFunc1(x) for x in X])
clustering = Clustering(X, y)

#create particles
particles = []
pop_size = 10
particle_size = 2
for i in range(pop_size):
    centroids = []
    for j in range(particle_size):
        centroids.append(np.array([ np.random.uniform(-1, 1, 1) for i in range(len(X[0]))]))
    particles.append(Particle(centroids, clustering.QuantizationError))
    
#create population
population = Population(particles, clustering.QuantizationError)

print("running...")
errors = []
# test
for i in range(10):
    population.update_pop()
    errors.append(clustering.QuantizationError(population.global_best))
# plt.plot(errors)
# plt.show()
