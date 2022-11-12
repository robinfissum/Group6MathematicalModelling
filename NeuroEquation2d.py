import numba as nb
import numpy as np
import PhysicalConstants2D
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, CartesianGrid, MemoryStorage, movie

raise DeprecationWarning('This file is deprecated. Zuzana is moving code into NeuroEq2D.py !')


class NeurotransmitterPDE(PDEBase):
    """Coupled equation for neurotransmitter-receptor reaction in 2D.
    n, r, rn denote the ScalarField (s) representing free neurotransmitter concentration,
    free receptor concentration and bound receptor-neurotransmitter pairs concentration,
    respectively.
    """

    def __init__(self, epsilon, epsilon_A, eta, eta_A, bc="auto_periodic_neumann"):
        super().__init__()
        self.epsilon = epsilon
        self.epsilon_A = epsilon_A
        self.eta = eta
        self.eta_A = eta_A
        self.bc = bc  # boundary condition
        self.initial_free_receptors = None

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        # Note: passing data=K below sets the initial value equal to K everywhere for the scalar field
        n = ScalarField(grid, data=0, label='$c_{N}$')
        for point in [[2, 2], [3, 2], [2, 3], [3, 3]]:
            n.insert(point=np.array(point), amount=40)

        r = ScalarField(grid, data=0, label='$c_{R}$')
        for point in [[4, 4], [5, 4], [4, 5], [5, 5]]:
            r.insert(point=np.array(point), amount=20)
        # Store the amount of neurotransmitters at t=0
        self.initial_free_receptors = r.integral

        # Concentration of bound receptor-neurotransmitter pairs is zero at t=0
        rn = ScalarField(grid, data=0, label='$c_{RN}$')
        return FieldCollection([n, r, rn])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        n, r, rn = state
        rhs = state.copy()
        epsilon = self.epsilon
        epsilon_A = self.epsilon_A
        eta = self.eta
        eta_A = self.eta_A

        if r.integral <= 0.5 * self.initial_free_receptors:
            print('Signal was sent at time t={t}')
            exit()
        


        rhs[0] = n.laplace(self.bc) - epsilon_A*r*n + eta_A * rn
        rhs[1] = -epsilon*r*n + eta*rn
        rhs[2] = epsilon*r*n - eta*rn
        return rhs

    def _make_pde_rhs_numba(self, state):
        """nunmba-compiled implementation of the PDE"""
        epsilon, epsilon_A = self.epsilon, self.epsilon_A
        eta, eta_A = self.eta, self.eta_A
        laplace = state.grid.make_operator("laplace", bc=self.bc)

        @nb.jit
        def pde_rhs(state_data, t):
            n = state_data[0]
            r = state_data[1]
            rn = state_data[2]

            rate = np.empty_like(state_data)
            rate[0] = laplace(n)-epsilon_A * r * n + eta_A * rn
            rate[1] = -epsilon * r * n + eta * rn
            rate[2] = epsilon * r * n - eta * rn
            return rate

        return pde_rhs


# Set zero Neumann boundary conditions (particles can't escape domain)
bc_x_left = {'derivative': 0}
bc_x_right = {'derivative': 0}
bc_y_left = {'derivative': 0}
bc_y_right = {'derivative': 0}
bc_x = [bc_x_left, bc_x_right]
bc_y = [bc_y_left, bc_y_right]
bc = [bc_x, bc_y]

# Create the grid
interval_x = (0, 40)
interval_y = (0, 40)
rectangular_domain = [interval_x, interval_y]
grid_points_per_dimension = [40, 40]
grid = CartesianGrid(bounds=rectangular_domain, shape=grid_points_per_dimension)


# # Change the parameteres to trivial
# epsilon = 1
# epsilon_A = 1
# eta = 1
# eta_A = 1
time_max = 2
dt = 0.01

# Create the PDE
# state represents the scalar fields defined on the grid
eq = NeurotransmitterPDE(epsilon=epsilon, epsilon_A=epsilon_A, eta=eta, eta_A=eta_A, bc=bc)
state = eq.get_initial_state(grid)

# Simulate the PDE and plot its evolution
storage = MemoryStorage()
sol = eq.solve(state, t_range=time_max, dt=dt, tracker=["progress", storage.tracker(0.01)], method='implicit')
movie(storage=storage, filename='./Animations/NeuroEvolution2D.gif')

# Get the data at all time steps of all concentrations
for time, field in storage.items():
    # Fetch concentrations objects. To get the values as numpy.ndarray , replace by field[i].data
    concentration_N = field[0]
    concentration_R = field[1]
    concentration_RN = field[2]

