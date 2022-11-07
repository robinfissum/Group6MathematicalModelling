import math
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from pde import CartesianGrid, DiffusionPDE, ScalarField, MemoryStorage, movie


# Define the grid parameters
interval_x = (0, 1)
interval_y = (0, 1)
rectangular_domain = [interval_x, interval_y]
grid_points_per_dimension = [50, 50]

# Generate the grid
grid = CartesianGrid(bounds=rectangular_domain, shape=grid_points_per_dimension)

# Generate initial condition
# state represents the scalar field defined on the grid
state = ScalarField(grid, label='concentration')

'''Put values inside the grid. If
rectangular_domain = [(0,1), (0,1)] and grid_points_per_dimension = [10, 10],
then typing state.insert(point=np.array([0, 0]), amount=1)
will result in the (0,0)-th element having the the value 100, since the (0,0)-th
volume element has Vol=0.1*0.1=0.01, and so 100*Vol = 1 
'''

state.insert(point=np.array([0.5, 0.5]), amount=1)

# state.data <<< to get current concentration

# Set zero Neumann boundary conditions (particles can't escape domain)
bc_x_left = {'derivative': 0}
bc_x_right = {'derivative': 0}
bc_y_left = {'derivative': 0}
bc_y_right = {'derivative': 0}
bc_x = [bc_x_left, bc_x_right]
bc_y = [bc_y_left, bc_y_right]
bc = [bc_x, bc_y]

# To keep track of values at each timestep.
storage = MemoryStorage()

# Define the PDE
diffusivity = 0.1
eq = DiffusionPDE(diffusivity=diffusivity, bc=bc)

# Solve the PDE over the interval [0,t_range] with temporal step size dt
# Store the values at each time step in the 'storage' -object.
t_max = 10
dt = 0.01
result = eq.solve(state, t_range=t_max, dt=dt, tracker=['progress', storage.tracker(interval=dt)])

#for time, field in storage.items():
#    print(f't={time}, field={field.magnitude}')

# For some reason only stores one frame
movie(storage=storage, filename='./Animations/evolution.gif')

# To get the final state as numpy ndarray, replace the line above by
# result = eq.solve(state, t_range=1, dt=0.01, ret_info=True)[0].data

# result.plot(cmap='magma')


