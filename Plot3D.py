import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from matplotlib.colors import ListedColormap
from time import perf_counter
import numpy as np
import imageio
import math
import os
import glob

# Choose backend. 'TkAgg', 'QtAgg' or 'GTKAgg' may be available options.
matplotlib.use('QtAgg')
'''
The purpose of this script is to plot the concentrations in 3D.
This is simply because it seems hard to make the py-pde plotting tools handle this
(the z-dimension is lost somewhere between the function- and method calls)
The outline of the code is taken from: 
https://github.com/robinfissum/Heat-Modelling-Using-Finite-Differences/blob/master/plots_and_gifs.py
'''


PLOT_FOLDER = 'Animations/'


class Plotter3D:
    def __init__(self, length, width, height, data_shape, points_per_dim='all'):
        """
        Args:
            length (float): Geometric length of domain in x-direction
            width (float): Geometric length of domain in y-direction
            height (float): Geometric length of domain in z-direction
            data_shape (list/tuple): Number of gridpoints in x, y and z-directions
            points_per_dim (int or 'all'): Determines how many grid points to plot in the x,y and z-directions.
                if 'all' is passed, then every grid point will be plotted.
        """
        self.length = length
        self.width = width
        self.height = height
        self.x_num, self.y_num, self.z_num = data_shape
        self.grid_shape = (self.x_num, self.y_num, self.z_num)
        self.points_per_dim = points_per_dim
        self.n1 = self.n2 = self.n3 = None
        self.x_skip_factor = self.y_skip_factor = self.z_skip_factor = None
        self.idxs = [[], [], []]

        # Set the 'skip-factors'. These determine how to skip indices we don't want to plot
        # in the case that an integer value of 'points_per_dim' has been specified.
        if isinstance(points_per_dim, int):
            if not math.gcd(self.x_num, self.y_num, self.z_num) % points_per_dim == 0:
                raise ValueError("Choose a 'points_per_dim' dividing each dimension of array.")
            else:
                n = self.points_per_dim
                self.x_skip_factor = self.x_num // n
                self.y_skip_factor = self.y_num // n
                self.z_skip_factor = self.z_num // n
                self.n1 = self.n2 = self.n3 = n
        elif isinstance(points_per_dim, str):
            if points_per_dim != 'all':
                raise ValueError(f'Please choose a valid "points per dim". Points per dim was {points_per_dim}')
            else:
                self.x_skip_factor = self.y_skip_factor = self.z_skip_factor = 1
                self.n1 = self.x_num
                self.n2 = self.y_num
                self.n3 = self.z_num

        # Compute the geometric coordinates of which points to plot
        for x in range(self.n1):
            for y in range(self.n2):
                for z in range(self.n3):
                    self.idxs[0].append(length/(self.x_num-1) * self.x_skip_factor * x)
                    self.idxs[1].append(width/(self.y_num-1) * self.y_skip_factor * y)
                    self.idxs[2].append(height/(self.z_num-1) * self.z_skip_factor * z)

        # Store the maximal geometric x,y and z values so that we can speed up plotting without the plot rescaling
        self.x_max = length/(self.x_num-1) * self.x_skip_factor * (self.n1 - 1)
        self.y_max = width/(self.y_num-1) * self.y_skip_factor * (self.n2 - 1)
        self.z_max = height/(self.z_num-1) * self.z_skip_factor * (self.n3 - 1)

    def save_plot(self, c_n: np.ndarray, c_r: np.ndarray, c_rn: np.ndarray,
                  time, filename='placeholder', rotate=False, elevate=False):
        """
        c_n, c_r, c_rn: numpy.ndarray, whose values are to be plotted in 3D
        time: (float). Time for which the concentrations are considered.
        filename: (string, OPTIONAL). Name of plot when saved in Animations/
        rotate: (integer, OPTIONAL). Specifies counterclockwise rotation of plots.
        elevate (integer, OPTIONAL). Specifies how much the plot is tilted (i.e. seen from above)
        returns: Nothing, but has the side effect of storing the plot as a png in the PLOT_FOLDER
        """

        # Check that array dimensions are compatible
        if not c_n.shape == c_r.shape == c_rn.shape == self.grid_shape:
            raise ValueError('Incompatible dimensions passed in save_plot.py.')

        fig = plt.figure(figsize=plt.figaspect(1 / 6))

        # To make a plot, we have to flatten the data (i.e. make it 1-dimensional)
        print('Configuring plot dimensions.')
        x_skip_factor, y_skip_factor, z_skip_factor = self.x_skip_factor, self.y_skip_factor, self.z_skip_factor

        flat_c_n = c_n[::x_skip_factor, ::y_skip_factor, ::z_skip_factor].flatten()
        flat_c_r = c_r[::x_skip_factor, ::y_skip_factor, ::z_skip_factor].flatten()
        flat_c_rn = c_rn[::x_skip_factor, ::y_skip_factor, ::z_skip_factor].flatten()

        n_max, r_max, rn_max = np.amax(c_n), np.amax(c_r), np.amax(c_rn)

        stop = False
        if np.isnan(c_n).any():
            print(f'c_n has Nan value at time={time}')
            stop = True
        if np.isnan(c_r).any():
            print(f'c_r has Nan value at time={time}')
            stop = True
        if np.isnan(c_rn).any():
            print(f'c_rn has Nan value at time={time}')
            stop = True
        if stop:
            raise OverflowError('It never goes according to plan, does it?')

        if n_max <= 0:
            n_max = 1
        if r_max <= 0:
            r_max = 1
        if rn_max <= 0:
            rn_max = 1

        '''
        Transparency is determined by a function f:[0, M]->[0,1], where 
        M is the maximal value of the array. In our case, we want f(v) to be close to
        zero unless v is close to M. 
        '''
        def transparency_function(number):
            if number == 0.0:
                return 0
            else:
                return max(0.001, number)

        alpha1 = np.array([transparency_function(i / n_max) for i in flat_c_n])
        alpha2 = np.array([transparency_function(i / r_max) for i in flat_c_r])
        alpha3 = np.array([transparency_function(i / rn_max) for i in flat_c_rn])

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        print('Plotting concentrations.')
        ignore_transparency_bug = False
        if ignore_transparency_bug:
            '''
            This will not work because of an unresolved bug in matplotlib (which makes the plotted
            transparency(alpha values) depend on the distance from the 'camera'). See: 
            # https://github.com/matplotlib/matplotlib/issues/22861
            # https://github.com/matplotlib/matplotlib/pull/23085
            for more information. The problem appears to be in mplot3d.py
            '''
            ax1.scatter(xs=self.idxs[0], ys=self.idxs[1], zs=self.idxs[2], alpha=alpha1, color='red', depthshade=False)
            ax2.scatter(xs=self.idxs[0], ys=self.idxs[1], zs=self.idxs[2], alpha=alpha2, color='blue', depthshade=False)
            ax3.scatter(xs=self.idxs[0], ys=self.idxs[1], zs=self.idxs[2], alpha=alpha3, color='green', depthshade=False)
        else:
            num_steps = len(flat_c_n)
            # To overcome the bug above, we have to plot each point separately. It is therefore quite slow.
            for x, y, z, n, r, rn, a1, a2, a3, i in zip(self.idxs[0], self.idxs[1], self.idxs[2], flat_c_n, flat_c_r, flat_c_rn, alpha1, alpha2, alpha3, range(num_steps)):
                print(f'Configuring {i}-th point out of {num_steps} data points.')
                if n != 0:
                    ax1.scatter(xs=x, ys=y, zs=z, alpha=a1, color='red')
                if r != 0:
                    ax2.scatter(xs=x, ys=y, zs=z, alpha=a2, color='blue')
                if rn != 0:
                    ax3.scatter(xs=x, ys=y, zs=z, alpha=a3, color='green')

            # Plot corner points to prevent rescaling
            for x in {0, self.x_max}:
                for y in {0, self.y_max}:
                    for z in {0, self.z_max}:
                        ax1.scatter(xs=x, ys=y, zs=z, alpha=0)
                        ax2.scatter(xs=x, ys=y, zs=z, alpha=0)
                        ax3.scatter(xs=x, ys=y, zs=z, alpha=0)

        ax1.update({'xlabel': 'X', 'ylabel': 'Y', 'zlabel': 'Z'})
        ax2.update({'xlabel': 'X', 'ylabel': 'Y', 'zlabel': 'Z'})
        ax3.update({'xlabel': 'X', 'ylabel': 'Y', 'zlabel': 'Z'})

        title = f't={round(time, 3)}'
        plt.title(title)

        if rotate:
            ax1.view_init(azim=rotate)
            ax2.view_init(azim=rotate)
            ax3.view_init(azim=rotate)
        if elevate:
            ax1.view_init(elev=elevate)
            ax2.view_init(elev=elevate)
            ax3.view_init(elev=elevate)
        print('Saving snapshot as a png.')
        plt.savefig(PLOT_FOLDER + filename + '.png')
        plt.close()
        print('Png Saved!')


def make_gif_from_local_png_files():
    """
    Run through the .png files in the PLOT_FOLDER directory and produce a .gif
    """
    filenames = glob.glob('./' + PLOT_FOLDER + '*.png')
    filenames.sort()
    with imageio.get_writer(PLOT_FOLDER + 'time_plot.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in filenames:
        os.remove(filename)


def give_sortable_name(time_steps, t):
    """
    This function should be used when you are generating many images that should be sorted alphanumerically
    time_steps: the total number names (i.e. files) you want to generate
    t: the idx of the current item you are at
    returns: (string) a name, with a certain number of 0s prefixed to make the names sortable
    """
    total_digits = math.floor(math.log10(time_steps))
    used_digits = 0 if t == 0 else math.floor(math.log10(t))
    return (total_digits - used_digits)*'0' + str(t)


def cartesian_product(*arrays):
    """
    Quickly compute the cartesian product of a collection of arrays and
    return the answer as a list of lists.
    Source: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
