import numba as nb
import numpy as np
import math
import PhysicalConstants
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, CartesianGrid, MemoryStorage, movie


class NeurotransmitterPDE(PDEBase):
    """Coupled equation for neurotransmitter-receptor reaction in 3D
    n, r, rn denote the ScalarField (s) representing free neurotransmitter concentration,
    free receptor concentration and bound receptor-neurotransmitter pairs concentration,
    respectively, in dimensionless forms.
    """

    def __init__(self, t_max, bc, verbose=False):
        if bc is None:
            raise TypeError('Specify boundary condition.')
        super().__init__()
        self.bc = bc  # boundary condition
        self.t_max = t_max
        self.epsilon = PhysicalConstants.epsilon
        self.eta = PhysicalConstants.eta
        self.t0 = PhysicalConstants.t_0
        self.initial_free_dimensionless_neurotransmitters = PhysicalConstants.N_initial_dimensionless
        self.initial_free_dimensionless_receptors = PhysicalConstants.R_initial_dimensionless
        self.verbose = verbose  # print information if python implementation is used.

    def get_initial_state(self, grid):
        """prepare a useful initial state
        F = ScalarField(grid, data=0, label="") sets the value equal to zero everywhere
        F.insert(point, amount=7) sets the integrated value equal to 7 over the box containing the point 'point'
        inside the grid. Thus, point are the coordinate, and not the indices in the grid.
        Note: If you are using insert K times, then amount must be replaced by:
        # amount=self.initial_free_dimensionless_neurotransmitters / K
        """
        n = ScalarField(grid, data=0, label="Field $c_{N}$")
        n.insert(point=np.array([30, 30, 0.9]), amount=self.initial_free_dimensionless_neurotransmitters)

        r = ScalarField(grid, data=0, label="Field $c_{R}$")
        r.insert(point=np.array([30, 30, 0.9]), amount=self.initial_free_dimensionless_receptors)

        # Concentration of bound receptor-neurotransmitter pairs is zero at t=0
        rn = ScalarField(grid, data=0, label="Field $c_{RN}$")
        return FieldCollection([n, r, rn])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE
        This is not run unless the numba overload is below is commented out, and is significantly slower.
        """
        n, r, rn = state
        rhs = state.copy()
        epsilon = self.epsilon
        eta = self.eta
        if self.verbose and t > 0:
            print(f'Running. {math.floor(100 * t / self.t_max)}% complete.')
            print(f'amounts: N={n.integral}, R={r.integral},RN={rn.integral} tot={n.integral+r.integral+2*rn.integral}')

        rhs[0] = n.laplace(self.bc) - epsilon * r * n + eta * rn
        rhs[1] = -epsilon * r * n + eta * rn
        rhs[2] = epsilon * r * n - eta * rn
        return rhs

    # For some reason the numba implementation is unable to interpret the state
    def _make_pde_rhs_numba(self, state):
        """nunmba-compiled implementation of the PDE
        This is faster, since it complies the code into C++ before running.
        """
        epsilon = self.epsilon
        eta = self.eta
        laplace = state.grid.make_operator('laplace', bc=self.bc)

        @nb.jit
        def pde_rhs(state_data, t):
            n = state_data[0]
            r = state_data[1]
            rn = state_data[2]

            rate = np.empty_like(state_data)
            rate[0] = laplace(n)-epsilon * r * n + eta * rn
            rate[1] = -epsilon * r * n + eta * rn
            rate[2] = epsilon * r * n - eta * rn
            return rate

        return pde_rhs


def main():
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
    boundary_condition = [bc_x, bc_y, bc_z]

    '''Create the grid. Because of our scaling, we should have interval_z = (0, 1). 
    For the synaptic cleft, radius/height = 14.7. Our x and y dimensions should therefore be 
    at least 2 times the diameter of the axon, i.e. >= 30.  
    '''
    interval_x = (0, 60)
    interval_y = (0, 60)
    interval_z = (0, 1)
    rectangular_domain = [interval_x, interval_y, interval_z]
    grid_points_per_dimension = [60, 60, 20]
    grid = CartesianGrid(bounds=rectangular_domain, shape=grid_points_per_dimension)

    # Set the relevant parameters
    time_max_seconds = math.pow(10, -9)
    time_max_dimensionless = time_max_seconds / PhysicalConstants.t_0
    dt_seconds = math.pow(10, -13)
    dt_dimensionless = dt_seconds / PhysicalConstants.t_0
    print(f'\nNumber of time steps = {time_max_dimensionless / dt_dimensionless}')

    # Give an indication about how likely it is that the scheme is unstable.
    # Recall that the dimensionless diffusivity now equals 1.
    cfl_constant = dt_dimensionless * ((grid_points_per_dimension[0] / (interval_x[1] - interval_x[0])) ** 2 +
                                       (grid_points_per_dimension[1] / (interval_y[1] - interval_y[0])) ** 2 +
                                       (grid_points_per_dimension[2] / (interval_z[1] - interval_z[0])) ** 2)
    print(f'The CFL constant is {cfl_constant}')
    if cfl_constant >= 0.5:
        print('The scheme will be unstable if an explicit method is used!')
        print('The issue can be resolved by using SMALLER time step, or using LARGER spatial steps.')
    else:
        print('The scheme is likely stable.')

    # Create the PDE
    # state represents the scalar fields defined on the grid
    eq = NeurotransmitterPDE(t_max=time_max_dimensionless, bc=boundary_condition, verbose=False)
    system_state = eq.get_initial_state(grid)

    # Store data of the simulation at fixed (dimensionless) time intervals
    store_data_interval = dt_dimensionless

    # Simulate the PDE and plot its evolution
    storage = MemoryStorage()
    _ = eq.solve(system_state, t_range=time_max_dimensionless, dt=dt_dimensionless, tracker=["progress", storage.tracker(store_data_interval)], method='implicit')

    # Create an animation of the PDE. This takes quite a while.
    create_animation = True
    if create_animation:
        # Todo: Plot the damn thing in 3d.
        print('Creating animation.')
        anim_format = ['gif', 'mp4'][0]
        movie(storage=storage, filename=f'./Animations/NeuroEvolution3D.{anim_format}')

    # get dimensionless volume of volume elements for each point in the gird
    volume_box = interval_x[1]*interval_y[1]*interval_z[1]/np.prod(grid_points_per_dimension)

    # Get the data at all time steps of all concentrations
    for time, field in storage.items():
        # Fetch concentrations objects. To get the values as numpy.ndarray , replace by field[i].data
        concentration_N = field[0]
        concentration_R = field[1]
        concentration_RN = field[2]
        if volume_box * np.sum(concentration_R.data) < 0.5 * PhysicalConstants.R_initial_dimensionless:
            print(f'Signal was sent at dimensionless time t={time}')
            print(f'That is, the signal was sent after t={time * PhysicalConstants.t_0}seconds.')


if __name__ == '__main__':
    main()
