import math

'''
This script constants the physical and dimensionless constants that we 
need in our mathematical model.
'''

# Number of elementary particles in one mole
mole = 6.02214076 * math.pow(10, 23)

# Height of synaptic cleft in meters
h = 1.5 * math.pow(10, -8)
# Radius of synaptic cleft in meters
r = 2.2 * math.pow(10, -7)
# Area of synaptic cleft cross-section (Not used below)
area = math.pi * math.pow(r, 2)
# Volume of synaptic cleft in cube meters (Not used below)
V = math.pi * math.pow(r, 2) * h

# Number of free neurotransmitters at t=0
N_initial = 5000

# Number of free neurotransmitters at t=0, measured in moles
# Should equal 8.302695335869232e-21
A_N = N_initial / mole

# Density of receptors on the axon membrane in (number / m^2)
density_receptors = math.pow(10, 15)

# Total number of receptors on the axon membrane = density * area
# Should be equal to 152.05308443374597
R_initial = density_receptors * math.pi * math.pow(r, 2)

# Total number of receptors on the axon membrane, measured in moles
# Should be equal to 2.524900869865187e-22
A_R = R_initial / mole

# Total number of free neurotransmitter plus total number of free receptors at t=0, measured in moles
# Should be equal to 8.555185422855751e-21
A_R_N = A_R + A_N

# Diffusivity of neurotransmitters in m^2/s 
alpha_N = 8 * math.pow(10, -7)

# Reaction constant k_{1} in L/(mol s)
k1_ignore = 4 * math.pow(10, 6)

# Reaction constant k_{1} in m^3/(mol s)
# (We don't like Liters, 1L=(m/10)^3=(1/1000)m^{3})
k1 = k1_ignore * math.pow(10, -3)

# Reaction constant k_{-1} in s^{-1}
k_1 = 5

# Dimensionless constant epsilon
# Should be equal to 4.219834862960392e-06
epsilon = k1 * A_R_N * h/(alpha_N * math.pi * math.pow(r, 2))

# Dimensionless constant eta
# Should be equal to 1.4062500000000004e-09
eta = k_1 * math.pow(h, 2) / alpha_N

# Timescale in seconds
# Should be equal to 2.812500000000001e-10
t_0 = math.pow(h, 2)/alpha_N

# Initial amount of dimensionless free neurotransmitters (i.e. volume integral over dimensionless cleft)
# Should be equal to 655.8467811735285
N_initial_dimensionless = A_N * math.pi * math.pow(r, 2) / (A_R_N * math.pow(h, 2))

# Initial amount of dimensionless free receptors (i.e. volume integral over dimensionless cleft)
# Should be equal to 19.944705198675816
R_initial_dimensionless = A_R * math.pi * math.pow(r, 2) / (A_R_N * math.pow(h, 2))

# To see the value of all variables, run:
# print(globals())
