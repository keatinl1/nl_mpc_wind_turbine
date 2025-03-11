'''
Defining the variables for different wind turbine models
Jonkman has pretty extensive docuentation on the variables
so that's what I'm using

Jonkman:
Data taken from: https://www.nrel.gov/docs/fy09osti/38060.pdf, pg6
more clearly presented: https://www.tara.tcd.ie/handle/2262/83163, pg75
this is the best i think as there are constraints given in the pdf..

'''

class Jonkman:
    def __init__(self):
        # State constraints
        self.max_Omega   = 1.2670    # rad/s
        self.max_theta   = 1.5708    # rad
        self.max_Qg      = 47402.91  # N*m

        # Input constraints
        self.max_pitch_rate  = 0.139626  # rad/s
        self.max_torque_rate = 15000.0   # N*m/s

        # Turbine characteristics
        self.radius = 61.5 # length of a single blade = radius of rotor
        self.moment_o_inertia = 11776047.0*3 # think this is the correct value as MoI given is for one blade and this is 3
        
        # Environment 
        self.wind_speed = 2.50
