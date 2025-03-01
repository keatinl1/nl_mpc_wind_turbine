'''
Defining the variables for different wind turbine models
Jonkman has pretty extensive docuentation on the variables
so that's what I'm using

Jonkman:
    Data taken from: https://www.nrel.gov/docs/fy09osti/38060.pdf, pg6
    more clearly presented: https://www.tara.tcd.ie/handle/2262/83163, pg75
    this is the best i think as there are constraints given in the pdf..

OdehJonkman:
    Data taken from: https://onlinelibrary.wiley.com/doi/epdf/10.1002/we.2853, pg990

Staino:
    PhD thesis: https://www.tara.tcd.ie/handle/2262/83163, pg172
    No constraints are given in the thesis, so they're just set to infinity

'''

class Jonkman:
    def __init__(self):
        # State constraints
        self.max_Omega   = 1.267     # rad/s
        self.max_theta   = 1.5708    # rad
        self.max_Qg      = 47402.91  # N*m

        # Input constraints
        self.max_pitch_rate  = 0.139626  # rad/s
        self.max_torque_rate = 15000.0   # N*m/s

        # Turbine characteristics
        self.radius = 61.5
        self.moment_o_inertia = 11776047.0
        # self.moment_o_inertia = 115.926E3
        # self.moment_o_inertia = 4.37e7 # ?. https://forums.nrel.gov/t/rotor-and-nacelle-mass-moment-of-inertia-tensors/2015/3
        # self.moment_o_inertia = 11776047.0*3 # think this is the correct value
        
        # Environment 
        self.wind_speed = 5.0       

#*********************************************************************************************************************
#* WILL LIKELY BE REMOVED 
#*********************************************************************************************************************

class OdehJonkman:
    def __init__(self):
        # state constraints
        self.max_Omega   = 1.267     # rad/s
        self.max_theta   = 1.5708    # rad
        self.max_Qg      = 47402.91  # N*m

        # input constraints
        self.max_pitch_rate  = 0.139626  # rad/s
        self.max_torque_rate = 15000.0   # N*m/s

        # turbine
        self.radius = 242.8/2
        self.moment_o_inertia = 7.01e8

        # environment 
        self.wind_speed = 3.0        

class Staino:    
    def __init__(self, **kwargs):
        # state constraints
        self.max_Omega = float('inf')   # rad/s
        self.max_theta = float('inf')   # rad
        self.max_Qg    = float('inf')   # N*m

        # input constraints
        self.max_pitch_rate  = float('inf') # rad/s
        self.max_torque_rate = float('inf') # N*m/s

        # turbine
        self.moment_o_inertia = 4747209 # ? https://forums.nrel.gov/t/rotor-and-nacelle-mass-moment-of-inertia-tensors/2015/3
        self.radius = 40

        # environment
        self.wind_speed = 12.0
