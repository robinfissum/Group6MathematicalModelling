# Group6MathematicalModelling

Prerequisites for running the code:

- You must have python installed (v.3.6 or later should suffice)

- If you want to generate plots using the movie() function from py-pde, 
you need to have FFmpeg installed and added to the system PATH environment variable.
(Internally the Movie class uses matplotlib.animation.FFMpegWriter., and it needs
to be added to PATH so that matplotlib can find it.)
To install it on Windows, you may follow the guide provided here:
https://www.wikihow.com/Install-FFmpeg-on-Windows

- In Plot3d.py, near the top, we have the line:  matplotlib.use('QtAgg')
This may not work unless you have PyQt6 installed. There are two possible fixes:
(1) Change backend: You may e.g. try to replace the line by matplotlib.use('TkAgg') or matplotlib.use('GTKAgg')
This should work because Tkinter comes with python by default.
(2) Alternatively, you can download PyQt6 from: https://www.riverbankcomputing.com/static/Docs/PyQt6/installation.html
(3), or another python backend, see: https://matplotlib.org/stable/users/explain/backends.html

### Overview of the code
The code utilizes the python package py-pde to solve systems of coupled PDEs.
The py-pde documentation can be found at https://py-pde.readthedocs.io/en/latest/getting_started.html
Several of the PDE solvers in the various files are modified versions of the example code found 
here: https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_brusselator_class.html#sphx-glr-examples-gallery-pde-brusselator-class-py

- PhysicalConstants.py and PhysicalConstants2d.py contain physical and dimensionless
constants needed in the simulations. 

- NeuroEquation3d.py contains code for simulating diffusion of neurotransmitters and reaction with 
receptors in 3D. 

- Plot3D.py contains code for generating a 3D gif of the evolution of the system in NeuroEquation3d.py

- 2d_solution.py contains code for simulating diffusion of neurotransmitters and reaction with 
receptors in 2d.

- ParticleSimOO.py contains code for modelling diffusion of neurotransmitters 
as a Brownian motion particle simulation (no PDEs here), as well as a probabilistic 
reaction with receptors (also modelled as discrete particles)

- transporters.ipynb contains code for simulating in 2d the PDE system 
between Neurotransmitters, Receptors, bound receptor-neurotransmitter-pairs, 
transporters, bound transporter-neurotransmitter-pairs and inactive naurotransmitters, 
with a flow term for the neurotransmitters. 

###Regarding questions about the code

You may contact the authors on 
- robin-fissum@hotmail.com
- robinol@stud.ntnu.no