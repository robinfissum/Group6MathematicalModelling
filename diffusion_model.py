import math
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


class SynapticCleft:
    def __init__(self, dx, dy, dz, dt, time_length, domain_dimensions: tuple):

        # Initialize steps sizes in x, y and z-directions
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.time_length = time_length

        # Get the dimensions of the domain
        self.omega_length = domain_dimensions[0]
        self.omega_width = domain_dimensions[1]
        self.omega_height = domain_dimensions[2]

        # Number of steps in x,y, and z-directions 
        self.length_steps = int(self.omega_length / self.dx) + 1
        self.width_steps = int(self.omega_width / self.dy) + 1
        self.height_steps = int(self.omega_height / self.dz) + 1
        self.time_steps = int(self.time_length / self.dt) + 1

        # Store concentrations in 3-dimensional numpy arrays
        self.c_N = None
        self.c_R = None
        self.c_RN = None 

    def set_initial_concentrations(self, initial_N, initial_R):
        # take in some initial data and set the initial concentrations
        # c_RN is zero at the beginning (no bound receptors)
        # Todo: Take functions or hard-coded matrices as input
        self.c_N = np.ones((self.length_steps, self.width_steps, self.height_steps))
        self.c_R = np.ones((self.length_steps, self.width_steps, self.height_steps))
        self.c_RN = np.zeros((self.length_steps, self.width_steps, self.height_steps))
        









