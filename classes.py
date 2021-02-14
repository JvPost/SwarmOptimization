# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:42:08 2021

@author: Glenn Viveen
"""
import numpy as np

# Global fitness function. Only applicable for Exercise 1.
def fitness(x):
    return np.sum(-x*np.sin(np.sqrt(np.abs(x))))

# Particle class
class Particle:
    def __init__(self, position, personal_best=None, velocity=None, bounds=None):
        """
        

        Parameters
        ----------
        position : 1-dimensional array-like
            list of coordinate values for the particle
        personal_best : 1-dimensional array-like, optional
            list of coordinate values for the best location this particular particle has visited. 
            By default this is set to the same coordinates given in position.
        velocity : 1-dimensional array-like, optional
            list of velocity values for the particle. The default is to initialize all velocities to 0.
        bounds : array-like, optional
            Array of size n x 2 where n is the dimensionality of the particle space. 
            The first and second column should contain the lower and higher bounds for each dimension respectively. 
            The default is no bounds.

        Returns
        -------
        None.

        """
        self.x = np.array(position) # current position
        self.bounds = bounds # bounds to which the particle is bound
        
        if personal_best is not None:
            self.best = personal_best # track individual best location visited
        else:
            self.best = self.x
        if velocity is not None:
            self.v = np.array(velocity) # track own velocity
        else:
            self.v = np.zeros_like(self.x)

    def fitness(self):
        """
        Returns the fitness of the particle at its current position

        Returns
        -------
        fitness
            The fitness of the particle at its current position.

        """
        return fitness(self.x)

    def update_self(self, global_best, r1=0.5, r2=0.5, alpha_1=1, alpha_2=1, omega=1, problem_type="min"):
        """
        Update the velocity, position, and individual best per the PSO algorithm.

        Parameters
        ----------
        global_best : 1-dimensional array-like
            coordinates of the best location found in the population so far.
        r1 : scalar, optional
            random scalar coefficient in the PSO update rule. The default is 0.5.
        r2 : scalar, optional
            random scalar coefficient in the PSO update rule. The default is 0.5.
        alpha_1 : scalar, optional
            weight coefficient for the individual best. The default is 1.
        alpha_2 : scalar, optional
            weight coefficient for the global best. The default is 1.
        omega : scalar, optional
            inertia coefficient. The default is 1.
        problem_type : string, optional
            Ought to be either "min" or "max". 
            Defines whether the optimization problem is a maximaization or minimization problem. 
            The default is "min".

        Returns
        -------
        x : 1-dimensional array-like
            New position.
        v : 1-dimensional array-like
            New velocity.
        fitness : scalar
            Fitness of particle at new position.

        """
        self.v = omega*self.v + alpha_1*r1*(self.best-self.x) + alpha_2*r2*(global_best - self.x) # Update velocity
        
        if self.bounds is not None:
            self.x = np.min((np.max((self.x+self.v, self.bounds[:,0]), axis=0), self.bounds[:, 1]), axis=0) # Update position with bounds
        else:
            self.x = self.x+self.v # Update position without bounds
        
        # Check if new personal best is attained
        if (self.fitness() > fitness(self.best)) and (problem_type=="max"):
            self.best = self.x 
        if (self.fitness() < fitness(self.best)) and (problem_type=="min"):
            self.best = self.x
        
        return self.x, self.v, self.fitness()
    
class Population:
    def __init__(self, particles, global_best=None, problem_type="min"):
        """
        Class to hold an entire population of Particles for optimization.
        
        Parameters
        ----------
        particles : array-like
            list of particles in the population. Should be instances of class Particle.
        global_best : 1-dimensional array-like, optional
            coordinate values for best visited location so far. 
            Default is to initialize to the best current location in the list of particles.
        problem_type : string, optional
            Ought to be either "min" or "max". 
            Defines whether the optimization problem is a maximaization or minimization problem. 
            The default is "min".
        """
        
        self.particles = particles # Define what particles are in the population.
        self.problem_type = problem_type # Define whether it is a maximization or minimization problem.
        
        if not(self.problem_type=="max" or self.problem_type=="min"):
            raise ValueError("problem_type should either 'min' or 'max'.")
        
        # Find global best if not defined manually
        if global_best is not None:
            self.global_best = global_best
        else:
            if self.problem_type=="min":
                self.global_best = self.particles[np.argmin([p.fitness() for p in self.particles])].x
            elif self.problem_type=="max":
                self.global_best = self.particles[np.argmax([p.fitness() for p in self.particles])].x
                
    def update_pop(self, **kwargs):
        """
        Update the location, velocity, and individual best for all particles and the global best as per the PSO algorithm.

        Parameters
        ----------
        **kwargs
            All arguments are passed to the update_self function of the Particle instances.

        Returns
        -------
        None.

        """
        if self.problem_type=="min":
            best_arg = np.argmin([p.update_self(self.global_best, problem_type=self.problem_type, **kwargs)[2] for p in self.particles])
        elif self.problem_type=="max":
            best_arg = np.argmax([p.update_self(self.global_best, problem_type=self.problem_type, **kwargs)[2] for p in self.particles])
            
        if (self.particles[best_arg].fitness() > fitness(self.global_best)) and (self.problem_type=="max"):
            self.global_best = self.particles[best_arg].fitness()
        elif (self.particles[best_arg].fitness() < fitness(self.global_best)) and (self.problem_type=="min"):
            self.global_best = self.particles[best_arg].fitness()
            
    def return_fitnesses(self):
        return {"particle {}".format(n):p.fitness() for n,p in enumerate(self.particles)}
    
    def return_positions(self):
        return {"particle {}".format(n):p.x for n,p in enumerate(self.particles)}
    
    def return_velocities(self):
        return {"particle {}".format(n):p.v for n,p in enumerate(self.particles)}