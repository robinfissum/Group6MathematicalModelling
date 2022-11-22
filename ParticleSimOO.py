import math
import time
import numpy as np
from numpy.linalg import norm
from numpy.random import normal
import random
import matplotlib.pyplot as plt
from PhysicalConstants import h, alpha_N


class Neurotransmitter:
    def __init__(self, pos: np.array):
        self.location = pos
        self.is_bound = False


class Receptor:
    def __init__(self, pos: np.array):
        self.location = pos
        self.is_free = True


class ParticleSim:
    def __init__(self, x_int: tuple, y_int: tuple, z_int: tuple, dt, neurotransmitters, receptors, reaction_radius):
        """
        x_int, y_int, z_int (tuples) specifies the domain
        neurotransmitters (list of numpy arrays)
        receptors (list of numpy arrays)
        """
        self.x_min, self.x_max = x_int
        self.y_min, self.y_max = y_int
        self.z_min, self.z_max = z_int
        self.transmitters = [Neurotransmitter(pos) for pos in neurotransmitters]
        self.receptors = [Receptor(pos) for pos in receptors]
        self.reaction_radius = reaction_radius
        self.dt = dt
        # Standard deviation of normally distributed transition function
        self.std_dev = math.sqrt(2*alpha_N*dt/3)/h
        self.number_bound_receptors = 0

    def in_domain(self, point: np.array):
        if self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max and self.z_min <= point[2] <= self.z_max:
            return True
        else:
            return False

    def update(self):
        for n in self.transmitters:
            if n.is_bound:
                # Probability of unbinding is set to 0.1% [Realistic value?]
                should_unbind = random.random() > 0.999
                if should_unbind:
                    n.is_bound = False
                    self.number_bound_receptors -= 1
                    # Unbind the correct receptor
                    for rec in self.receptors:
                        if not (n.location-rec.location).any():
                            rec.is_free = True
            else:
                # See if n is close to a free receptor.
                for r in self.receptors:
                    if r.is_free and norm(n.location-r.location) <= self.reaction_radius:
                        '''
                        Probability of reaction decreases linearly with the distance from the receptor. 
                        If d is the Euclidean distance between the receptor and the neurotransmitter, then 
                        the probability of reaction is: 1 - d/R  if 0<=d<=R and zero if d>R, where
                        R=self.reaction_radius is the "effective reaction radius".
                        This may not be very realistic.                    
                        '''
                        random_float = random.uniform(0.0, 1.0)
                        if random_float <= 1-norm(n.location-r.location) / self.reaction_radius:
                            print('Binding occurred!')
                            n.location = np.copy(r.location)
                            n.is_bound = True
                            r.is_free = False
                            self.number_bound_receptors += 1
                            break

            if not n.is_bound:
                # If n is not bound at this point in time, it will take a "Brownian step"
                # This is done using a normal distribution for each component
                dxs = normal(loc=0, scale=self.std_dev, size=3)
                while not self.in_domain(n.location + dxs):
                    dxs = normal(loc=0, scale=self.std_dev, size=3)
                n.location += dxs

    def has_bound_pair(self):
        """
        Return True if some receptors is bound. Otherwise, return False.
        """
        return any(rec.is_free is False for rec in self.receptors)

    def get_fraction_bound(self):
        return self.number_bound_receptors / len(self.receptors)


def main():
    t1 = time.time()
    # Set up the domain
    x_min, x_max = 0, 60
    y_min, y_max = 0, 60
    z_min, z_max = 0, 1
    x_interval = (x_min, x_max)
    y_interval = (y_min, y_max)
    z_interval = (z_min, z_max)
    dt = math.pow(10, -6)

    # Define initial placement of neurotransmitters (there are 5000=10*50)
    neuro = []
    for r in np.linspace(0, 14.7, 100):
        for angle in np.linspace(0, math.tau, 51)[0:-1]:
            neuro.append(np.array([30+r*math.cos(angle), 30+r*math.sin(angle), z_max]))

    # Define initial placement of free receptors, set reaction radius(there are 152=8*19)
    receptors = []
    for r in np.linspace(0, 14.7, 8):
        for angle in np.linspace(0, math.tau, 20)[0:-1]:
            receptors.append(np.array([30+r*math.cos(angle), 30+r*math.sin(angle), z_min]))

    reaction_radius = 1/5

    sim = ParticleSim(x_int=x_interval, y_int=y_interval, z_int=z_interval, neurotransmitters=neuro,
                      receptors=receptors, reaction_radius=reaction_radius, dt=dt)

    time_steps = 100

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the trajectory of a neurotransmitter
    xs1 = [sim.transmitters[0].location[0]]
    ys1 = [sim.transmitters[0].location[1]]
    zs1 = [sim.transmitters[0].location[2]]

    signal_sent_timestep = None
    # Run the simulation
    for t in range(time_steps):
        sim.update()
        print(f'Time step {t} of {time_steps} completed. Fraction R bound={sim.get_fraction_bound()}')
        if not signal_sent_timestep and sim.get_fraction_bound() >= 0.5:
            signal_sent_timestep = t
        xs1.append(sim.transmitters[0].location[0])
        ys1.append(sim.transmitters[0].location[1])
        zs1.append(sim.transmitters[0].location[2])

    print(f'Time taken for simulation: {time.time() - t1}s.')
    # Plot the line between the positions of the particles at different time steps
    for i in range(time_steps):
        ax.plot3D([xs1[i], xs1[i + 1]], [ys1[i], ys1[i + 1]], [zs1[i], zs1[i + 1]], color='red')

    # Plot receptors
    for rec in receptors:
        ax.scatter(xs=rec[0], ys=rec[1], zs=rec[2], color='green')

    # Plot corner points of the domain to keep view fixed.
    for x in x_interval:
        for y in y_interval:
            for z in z_interval:
                ax.scatter(xs=x, ys=y, zs=z, alpha=0)

    ax.scatter(xs=xs1, ys=ys1, zs=zs1, color='blue', depthshade=False)
    ax.update({'xlabel': 'X', 'ylabel': 'Y', 'zlabel': 'Z'})
    plt.show()
    print(f'The signal was sent after t={signal_sent_timestep}')

    print(f'Total number of time steps:{time_steps}.')


if __name__ == '__main__':
    main()
