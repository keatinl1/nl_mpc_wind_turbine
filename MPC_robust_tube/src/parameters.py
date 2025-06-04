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
        self.max_theta   = 90.0      # deg
        self.max_Qg      = 47.40291  # kN*m

        # Input constraints
        self.max_pitch_rate  = 8         # deg/s
        self.max_torque_rate = 15.00   # kN*m/s

        # Turbine characteristics
        self.radius = 61.5 # length of a single blade = radius of rotor
        self.moment_o_inertia = 11776047.0*3 # think this is the correct value as MoI given is for one blade and this is 3
        
        # Environment 
        self.air_density = 1.225
        # self.wind_speed = 3.0   # cut in wind speed
        # self.wind_speed = 5.5
        # self.wind_speed = 8.0
        self.wind_speed = 10.0
        # self.wind_speed = 25.0  # cut out wind speed
