import numba as nb
import numpy as np

from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid


class NeurotransmitterPDE(PDEBase):
    """Coupled equation for neurotransmitter-receptor reaction in 3D
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

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        # Note: passing data=1 below sets the initial value equal to 1 everywhere for the scalar field
        n = ScalarField(grid, data=0, label="Field $c_{N}$")
        n.insert(point=np.array([2, 2, 2]), amount=1000)

        r = ScalarField(grid, data=0, label="Field $c_{R}$")
        r.insert(point=np.array([10, 10, 10]), amount=1000)

        # Concentration of bound receptor-neurotransmitter pairs is zero at t=0
        rn = ScalarField(grid, data=0, label="Field $c_{RN}$")
        return FieldCollection([n, r, rn])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        n, r, rn = state
        rhs = state.copy()
        epsilon = self.epsilon
        epsilon_A = self.epsilon_A
        eta = self.eta
        eta_A = self.eta_A

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
bc_z_left = {'derivative': 0}
bc_z_right = {'derivative': 0}
bc_x = [bc_x_left, bc_x_right]
bc_y = [bc_y_left, bc_y_right]
bc_z = [bc_z_left, bc_z_right]
bc = [bc_x, bc_y, bc_z]


# initialize state
# Todo: Replace by generic grid.
grid = UnitGrid([64, 64, 64])
eq = NeurotransmitterPDE(epsilon=1, epsilon_A=1, eta=1, eta_A=1, bc=bc)
state = eq.get_initial_state(grid)

# simulate the pde
tracker = PlotTracker(interval=0.05, plot_args={"vmin": 0, "vmax": 5})
sol = eq.solve(state, t_range=20, dt=0.01, tracker=tracker)

# Todo: Plot the damn thing in 3d.