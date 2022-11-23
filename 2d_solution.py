import pde
from pde import FieldCollection
import numpy as np
import time
import math
import PhysicalConstants
import numpy as np
from Plot3D import cartesian_product

t1 = time.time()

# 2d grid:
x_min = -30
x_max = 30
y_min = -30
y_max = 30
x_interval = (x_min, x_max)
y_interval = (y_min, y_max)
x_num_gridpoints = 60
y_num_gridpoints = 60
gridpoints = [x_num_gridpoints, y_num_gridpoints]

# Define the grid
grid = pde.CartesianGrid([x_interval, y_interval], gridpoints)

# Get coordinates (DOES NOT EQUAL INDICES!) of grid nodes as a list
grid_nodes_coordinates = cartesian_product(*grid.axes_coords)

# Determine coordinates of pre/post synaptic sites(= the same) The dimensionless radius is r = 14.7
reaction_cite = []
for x, y in grid_nodes_coordinates:
    if np.linalg.norm(x=(x, y)) < 14.7:
        reaction_cite.append((x, y))


# Initial field for each concentration
# Look into change release site so there's a higher concentration in a smaller area, like in the pictures.
field_c_N = pde.ScalarField(grid, data=0, label="c_N field")
field_c_R = pde.ScalarField(grid, data=0, label="c_R field")
field_c_RN = pde.ScalarField(grid, data=0, label="c_RN field")

# Insert initial concentrations
for point in reaction_cite:
    field_c_N.insert(point=np.array(point), amount=PhysicalConstants.N_initial_dimensionless / len(reaction_cite))

for point in reaction_cite:
    field_c_R.insert(point=np.array(point), amount=PhysicalConstants.R_initial_dimensionless / len(reaction_cite))


'''
#Todo: This is for debugging if needed.
print(field_c_N.integral)
print(field_c_R.integral)
print(field_c_RN.integral)
print(np.unique(field_c_N.data, return_counts=True))
# print(np.unique(field_c_R.data, return_counts=True))
# print(np.unique(field_c_RN.data, return_counts=True))
# exit()
'''

state = FieldCollection([field_c_N, field_c_R, field_c_RN])

# Use same dimensionless parameters as for the 3D case.
eta = 0.00242
epsilon = 3000 * eta

eq = pde.PDE({
    "c_N": f"laplace(c_N) -{epsilon}*c_R*c_N + {eta}*c_RN",
    "c_R": f"-{epsilon}*c_R*c_N +{eta}*c_RN",
    "c_RN": f"{epsilon}*c_R*c_N -{eta}*c_RN",
    },
    bc='neumann')

storage = pde.MemoryStorage()

# Configure time parameters
time_max_seconds = 50 * math.pow(10, -7)
dt_seconds = math.pow(10, -7)
time_max_dimless = time_max_seconds / PhysicalConstants.t_0
dt_dimless = dt_seconds / PhysicalConstants.t_0

# Solve the equation and store intermediate states
result = eq.solve(state, t_range=time_max_dimless, dt=dt_dimless, tracker=["progress", storage.tracker(dt_dimless)])


box_volume = (x_max-x_min)*(y_max-y_min) / (x_num_gridpoints * y_num_gridpoints)

# Loop through the stored states.
for the_time, field in storage.items():
    # Fetch concentrations objects. To get their values as numpy.ndarray, replace them by field[i].data
    concentration_n = field[0]
    concentration_r = field[1]
    concentration_rn = field[2]

    # Check if half of the receptors have bounded to neurotransmitters
    if box_volume * np.sum(concentration_r.data) < 0.5 * PhysicalConstants.R_initial_dimensionless:
        print(f'Signal was sent at dimensionless time t={the_time}')
        print(f'That is, the signal was sent after t={the_time * PhysicalConstants.t_0}seconds.')
        break

# print(storage)
# print(type(storage))

print(f'Time taken for entire simulation={time.time()-t1}s')
pde.movie(storage, filename="./Animations/res2.mp4")