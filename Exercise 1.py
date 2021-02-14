# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:07:23 2021

@author: Glenn Viveen
"""

from classes import Particle, Population
import numpy as np

bounds = np.array([[-500, 500], [-500, 500]])
x1 = Particle((-400, -400), velocity=(-50, -50), bounds=bounds)
x2 = Particle((-410, -410), velocity=(-50, -50), bounds=bounds)
x3 = Particle((-415, -415), velocity=(-50, -50), bounds=bounds)

population = Population([x1, x2, x3], problem_type="max")