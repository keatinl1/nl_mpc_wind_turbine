class Jonkman:
    # Data taken from: https://www.nrel.gov/docs/fy09osti/38060.pdf, pg6
    # more clearly presented: https://www.tara.tcd.ie/handle/2262/83163, pg75

    def __init__(self, **kwargs):
        # state constraints
        self.max_Omega   = 1.267     # rad/s
        self.max_theta   = 1.5708    # rad
        self.max_Qg      = 47402.91  # N*m

        # input constraints
        self.max_pitch_rate  = 0.139626  # rad/s
        self.max_torque_rate = 15000.0   # N*m/s

        # model 
        self.wind_speed = 12.0

        self.radius = 61.5
        self.moment_o_inertia = 11776047.0
        # self.moment_o_inertia = 11776047.0*3.0
        # self.moment_o_inertia = 4.37e7 # ? https://forums.nrel.gov/t/rotor-and-nacelle-mass-moment-of-inertia-tensors/2015/3
        

class Staino:
    # https://www.tara.tcd.ie/handle/2262/83163
    def __init__(self, **kwargs):
        # state constraints
        self.max_Omega   = 1.267     # rad/s
        self.max_theta   = 1.5708    # rad
        self.max_Qg      = 47402.91  # N*m

        # input constraints
        self.max_pitch_rate  = 0.139626  # rad/s
        self.max_torque_rate = 15000.0   # N*m/s

        # model 
        self.wind_speed = 12.0

        self.moment_o_inertia = 4747209 # ? https://forums.nrel.gov/t/rotor-and-nacelle-mass-moment-of-inertia-tensors/2015/3
        self.radius = 40
