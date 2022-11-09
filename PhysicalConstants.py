import math

'''
This script constants the physical and dimensionless constant that we 
need in our mathematical model.
'''

# Number of elementary particles in one mole
mole = 6.02214076*math.pow(10, 23)

# height of synaptic cleft in meters 
h = 15 * math.pow(10, -9)
# radius of synaptic cleft in meters
r = 0.22 * math.pow(10, -6)
# area of synaptic cleft cross-section
area = math.pi * math.pow(r, 2)
# volume of synaptic cleft in cube meters
V = math.pi * math.pow(r, 2) * h

# initial neurotransmitter quantity
N_initial = 5000

# initial neurotransmitter quantity in mole
A_N = N_initial / mole

# density of receptors on the axon membrane in (number / m^2)
density_receptors = math.pow(10, 15)

# Total number of receptors on the axon membrane
# should be equal to 152.05308443374597
R_initial = density_receptors * math.pi * math.pow(r, 2)

# Total number of receptors on the axon membrane in mole
A_R = R_initial / mole

# Diffusivity of neurotransmitters in m^2/s 
alpha_N = 8 * math.pow(10, -7)

# reaction constant k_{1} in (mol / L)^{-1} s^{-1}
_k1 = 4 * math.pow(10, 6)

# reaction constant k_{1} in (mol / m^{3}) s^{-1}
# (We don't like Liters, 1L=(m/10)^3=(1/1000)m^{3})
k1 = 4 * math.pow(10, 3)

# reaction constant k_{-1} in s^{-1}
k_1 = 5

# dimensionless constant epsilon
# should be equal to 4.095294432922354e-06
epsilon = k1 * A_N *math.pow(h, 2) /(V * alpha_N)

# Dimensionless constant eta
# Should be equal to 1.4062500000000004e-09
eta = k_1 * math.pow(h, 2) / alpha_N

# other constants
A = A_R / A_N
epsilon_A = A * epsilon
eta_A = A * eta

# print all names and values
# print(globals())



