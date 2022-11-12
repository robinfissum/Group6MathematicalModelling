import numba as nb
import numpy as np
import math
import PhysicalConstants
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, CartesianGrid, MemoryStorage, movie
from Plot3D import Plotter3D, make_gif_from_local_png_files, give_sortable_name


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
        n = ScalarField(grid, data=0, label='$c_{N}$')
        presynaptic_terminal = []
        postsynaptic_terminal = []
        for i in range(60):
            for j in range(60):
                if math.pow(i-30, 2)+math.pow(j-30, 2) <= math.pow(14.7, 2):
                    presynaptic_terminal.append((i, j, 0.9))
                    postsynaptic_terminal.append((i, j, 0.1))

        for point in presynaptic_terminal:
            n.insert(point=np.array(point), amount=self.initial_free_dimensionless_neurotransmitters/len(presynaptic_terminal))

        r = ScalarField(grid, data=0, label='$c_{R}$')
        for point in postsynaptic_terminal:
            r.insert(point=np.array(point), amount=self.initial_free_dimensionless_receptors/len(postsynaptic_terminal))

        # Concentration of bound receptor-neurotransmitter pairs is zero at t=0
        rn = ScalarField(grid, data=0, label='$c_{RN}$')
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
            print(
                f'amounts: N={n.integral}, R={r.integral},RN={rn.integral} tot={n.integral + r.integral + 2 * rn.integral}')

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
            rate[0] = laplace(n) - epsilon * r * n + eta * rn
            rate[1] = -epsilon * r * n + eta * rn
            rate[2] = epsilon * r * n - eta * rn
            return rate

        return pde_rhs


def main():
    # Set zero Neumann boundary condition everywhere (particles can't escape domain)
    boundary_condition = {'derivative': 0}

    # Create the grid. Be sure to choose the components of 'num_gridpoints' such that they have a high gcd.
    interval_x = (0, 60)
    interval_y = (0, 60)
    interval_z = (0, 1)
    length_x, length_y, length_z = interval_x[1]-interval_x[0], interval_y[1]-interval_y[0], interval_z[1]-interval_z[0]
    rectangular_domain = [interval_x, interval_y, interval_z]
    num_gridpoints = [66, 66, 33]
    grid = CartesianGrid(bounds=rectangular_domain, shape=num_gridpoints)

    # Set the time interval and temporal stepsize
    time_max_seconds = 5 * math.pow(10, -13)
    dt_seconds = math.pow(10, -13)
    time_max_dimless = time_max_seconds / PhysicalConstants.t_0
    dt_dimless = dt_seconds / PhysicalConstants.t_0

    # If time_max_dimless is not divisible by dt_dimless, ignore the "remainder time"
    num_time_steps = math.floor(time_max_dimless / dt_dimless) + 1
    print(f'\nNumber of time steps = {num_time_steps}')

    # Give an indication about how likely it is that the scheme is unstable.
    # Recall that the dimensionless diffusivity now equals 1.
    cfl_constant = dt_dimless * ((num_gridpoints[0] / length_x) ** 2 +
                                 (num_gridpoints[1] / length_y) ** 2 +
                                 (num_gridpoints[2] / length_z) ** 2)
    print(f'The CFL constant is {round(cfl_constant,2)}...')
    if cfl_constant > 0.5:
        print('The scheme will be UNSTABLE if an explicit method is used!')
        print('The issue may be resolved by using a smaller time step.')
    else:
        print('The scheme is likely STABLE.')

    # Create the PDE
    # system_state represents the scalar fields defined on the grid
    eq = NeurotransmitterPDE(t_max=dt_dimless * math.floor(time_max_dimless / dt_dimless), bc=boundary_condition,
                             verbose=False)
    system_state = eq.get_initial_state(grid)

    # Store data of the simulation at fixed (dimensionless) time intervals
    store_data_interval = dt_dimless

    # Simulate the PDE and store its data
    storage = MemoryStorage()
    _ = eq.solve(system_state, t_range=dt_dimless * math.floor(time_max_dimless / dt_dimless), dt=dt_dimless,
                 tracker=['progress', storage.tracker(store_data_interval)], method='implicit')

    # Create a 3d animation of the PDE? This takes quite a while. If there are many gridpoints,
    # then it is recommended to not plot all of them. Thus, choose points_per_dim wisely.
    i_want_to_create_a_3d_animation = True
    points_per_dim = [math.gcd(*num_gridpoints), 'all'][0]
    plotter = Plotter3D(length=num_gridpoints[0], width=num_gridpoints[1], height=num_gridpoints[2],
                        data_shape=num_gridpoints, points_per_dim=points_per_dim)

    # Determine the volume of the boxes of the discretization. For reference,
    # see: https://py-pde.readthedocs.io/en/latest/manual/mathematical_basics.html#spatial-discretization
    box_volume = length_x * length_y * length_z / np.prod(num_gridpoints)

    # Get the data at all time steps of all concentrations
    for time, field in storage.items():
        # Fetch concentrations objects. To get their values as numpy.ndarray, replace them by field[i].data
        concentration_n = field[0]
        concentration_r = field[1]
        concentration_rn = field[2]

        # Check if half of the receptors have bounded to neurotransmitters
        if box_volume * np.sum(concentration_r.data) < 0.5 * PhysicalConstants.R_initial_dimensionless:
            print(f'Signal was sent at dimensionless time t={time}')
            print(f'That is, the signal was sent after t={time * PhysicalConstants.t_0}seconds.')

        # Make a .png snapshot of the molecule concentrations
        if i_want_to_create_a_3d_animation:
            plotter.save_plot(c_n=concentration_n.data, c_r=concentration_r.data, c_rn=concentration_rn.data, time=time,
                              filename=f'timestep{give_sortable_name(round(time_max_dimless / dt_dimless), round(time / dt_dimless))}',
                              rotate=False, elevate=False)

    if i_want_to_create_a_3d_animation:
        make_gif_from_local_png_files()


if __name__ == '__main__':
    main()
