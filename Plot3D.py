import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import imageio
import math
import os
import glob
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, CartesianGrid, MemoryStorage

'''
The purpose of this script is to plot the concentrations in 3D.
This is simply because it seems hard to make the py-pde plotting tools
take care of this (the z-dimension is lost somewhere between the function- and method calls)
The outline of the code is taken from: 
https://github.com/robinfissum/Heat-Modelling-Using-Finite-Differences/blob/master/plots_and_gifs.py
'''


PLOT_FOLDER = "Animations/"


def save_plot(c_n: np.ndarray, c_r: np.ndarray, c_rn: np.ndarray, time, length, width, height, points_per_dim, filename='placeholder', rotate=False, elevate=False, plane=None):
    """
    c_n, c_r, c_rn: numpy ndarray -s, whose values are to be plotted in 3D
    length: (number) length of the domain (x-direction)
    width: (number) width of the domain (y-direction)
    height: (number) height of the domain (z-direction)
    points_per_dim: (number or 'all'). If a number K is provided, then the plot will show K grid points in eaach
    dimension. If 'all' is provided, then all the grid points will be plotted. 
    filename: (string, OPTIONAL) name of plot when saved in animations/
    rotate: (integer, OPTIONAL) specifies counterclockwise rotation of plots.
    elevate (integer, OPTIONAL) specifies how much the plot is tilted (i.e. seen from above)
    returns: nothing, but has the side effect of storing the plot as a png in the PLOT_FOLDER
    """

    # Get the shape of the concentration arrays to be plotted.
    if not c_n.shape == c_r.shape == c_rn.shape:
        raise ValueError('Incompatible dimensions passed in save_plot.py.')
    else: 
        x_num, y_num, z_num = np.shape(c_n.data)

    if isinstance(points_per_dim, int):
        if not math.gcd(x_num, y_num, z_num) % points_per_dim == 0:
            raise ValueError("Problematic scaling. Choose 'points_per_dim' so that it divides each dimension of the concentraction arrays. ")
        else:
            n = points_per_dim
            x_skip_factor = x_num // n
            y_skip_factor = y_num // n
            z_skip_factor = z_num // n
            n1 = n2 = n3 = n
    elif isinstance(points_per_dim, str):
        if points_per_dim != 'all':
            raise ValueError(f'Please choose a valid "Points per dim". Points per dim was {points_per_dim}')
        else:
            x_skip_factor = y_skip_factor = z_skip_factor = 1
            n1 = x_num
            n2 = y_num
            n3 = z_num


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Set a color scheme for the plot
    # Todo: Can we do one for each concentration?
    scale_colors = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    

    # Create a scatter plot depending on the parameters passed.
    idxs = [[], [], []]
    temp = np.array([])
    # Todo: NB: These were just added
    temp2 = np.array([])
    temp3 = np.array([])

    for x in range(n1):
        for y in range(n2):
            for z in range(n3):
                x = x if not plane or (
                    plane and not "x" in plane) else plane["x"]
                y = y if not plane or (
                    plane and not "y" in plane) else plane["y"]
                z = z if not plane or (
                    plane and not "z" in plane) else plane["z"]

                idxs[0].append(length/(x_num-1) * x_skip_factor * x)
                idxs[1].append(width/(y_num-1) * y_skip_factor * y)
                idxs[2].append(height/(z_num-1) * z_skip_factor * z)
                temp = np.append(temp, c_n[x_skip_factor * x, y_skip_factor * y, z_skip_factor*z])
                # NB: These were just added 
                temp2 = np.append(temp2, c_r[x_skip_factor * x, y_skip_factor * y, z_skip_factor*z])
                temp3 = np.append(temp3, c_rn[x_skip_factor * x, y_skip_factor * y, z_skip_factor*z])

    im = ax.scatter(idxs[0], idxs[1], idxs[2], c=temp, marker='o', cmap=plt.cm.coolwarm, norm=scale_colors)
    im = ax.scatter(idxs[0], idxs[1], idxs[2], c=temp2, marker='o', cmap=plt.cm.coolwarm, norm=scale_colors)
    im = ax.scatter(idxs[0], idxs[1], idxs[2], c=temp3, marker='o', cmap=plt.cm.coolwarm, norm=scale_colors)

    # 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = f'Concentrations at time = {time}'
    plt.title(title)
    plt.colorbar(im)
    if rotate:
        ax.view_init(azim=rotate)
    if elevate:
        ax.view_init(elev=elevate)
    plt.savefig(PLOT_FOLDER + filename + ".png")
    plt.close()

# Ignore this: 
# To center the heat plot such that 0 degrees get no color, positive temperatures are red, and negative blue
# Todo: v_min and v_max should be set according to the highest and lowest temperature, possibly
# Todo: such that v_min = - v_max
# See: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
# divnorm = colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)



def turn_pngs_to_gif():
    """
    Run through the .png files in the PLOT_FOLDER directory and produce a gif. 
    """
    filenames = glob.glob("./" + PLOT_FOLDER + "*.png")
    filenames.sort()
    with imageio.get_writer(PLOT_FOLDER + "time_plot.gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in filenames:
        os.remove(filename)



def give_sortable_name(time_steps, t):
    """
    This functions should be used when you are generating many images that should be in sortable order alphanumerically
    timesteps: the total number names you want to generate
    t: the idx of the current item you are at
    returns: (string) a name, with a certain number of 0s prefixed to make the names sortable
    """
    total_digits = math.floor(math.log10(time_steps))
    used_digits = 0 if t == 0 else math.floor(math.log10(t))
    return (total_digits-used_digits)*"0" + str(t)

