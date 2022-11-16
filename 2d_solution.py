
import pde
from pde import FieldCollection, PDEBase, UnitGrid
import numpy as np

#2d grid:
grid = pde.CartesianGrid([[-10,10],[-10,10]], [100, 100])

#Initial field for each concentration
#Look into change release site so there's a higher conentration in a smaller area, like in the pictures.
field_c_N = pde.ScalarField.from_expression(grid, "x**2 + y**2 <= 2**2", label="c_N field")

field_c_R = pde.ScalarField.from_expression(grid, "x**2 + y**2 <= 2**2", label="c_R field")
field_c_R.data = field_c_R.data*0.1

field_c_RN = pde.ScalarField(grid, label="c_RN field")
print("")
print(np.unique(field_c_N.data, return_counts=True))
print(np.unique(field_c_R.data, return_counts=True))
print(np.unique(field_c_RN.data, return_counts=True))

state = FieldCollection([field_c_N, field_c_R, field_c_RN])

epsilon = 1 #3.6309e-5
eta = 1 #0.80666

eq = pde.PDE({
    "c_N" : f"laplace(c_N) -{epsilon}*c_R*c_N + {eta}*c_RN",
    "c_R" : f"-{epsilon}*c_R*c_N +{eta}*c_RN",
    "c_RN" : f"{epsilon}*c_R*c_N -{eta}*c_RN",
    })

storage = pde.MemoryStorage()

result = eq.solve(state, t_range=2, dt=1e-3, tracker=["progress", storage.tracker(1e-3)])


print(storage)
print(type(storage))
#pde.movie(storage, filename="./Animations/res2.mp4")