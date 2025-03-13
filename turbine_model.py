from acados_template import AcadosModel
from casadi import SX, vertcat, exp, pi


# Define/import some constants
from parameters import Jonkman
params = Jonkman()
Jt  = params.moment_o_inertia
rho = params.air_density
V   = params.wind_speed
R   = params.radius

# Reference for model equations:
# https://backend.orbit.dtu.dk/ws/portalfiles/portal/5832126/prod21318234687066.2070.pdf

def export_robot_model() -> AcadosModel:
    model_name = "turbine"

    # States
    Omega = SX.sym("Omega") # Rotor speed
    theta = SX.sym("theta") # Blade pitch angle
    Qg    = SX.sym("Qg")    # Generator torque
    x = vertcat(Omega, theta, Qg)

    # Inputs
    u1 = SX.sym("u1")  # Pitch rate
    u2 = SX.sym("u2")  # Generator torque rate
    u = vertcat(u1, u2)

    # Dynamics
    Omega_dot = SX.sym("Omega_dot")
    theta_dot = SX.sym("theta_dot")
    Qg_dot    = SX.sym("Qg_dot")
    xdot = vertcat(Omega_dot, theta_dot, Qg_dot)

    # updated Cp
    # is in DEGREES!!
    c1 = 0.5176
    c2 = 116
    c3 = 0.4
    c4 = 5
    c5 = 21
    c6 = 0.0068

    L = Omega * R / V
    Li = 1 / (1 / (L + 0.08 * theta) - 0.035 / (theta**3 + 1))
    Cp1 = c1 * (c2 / Li - c3 * theta - c4)  # Using scaled theta here
    Cp2 = exp(-c5 / Li)
    Cp3 = c6 * L
    Cp = Cp1 * Cp2 + Cp3

    Q = (0.5*rho*pi*(R**2)*(V**3)*Cp)/(Omega*1000)

    # Explicit system dynamics
    f_expl = vertcat((1000/Jt)*(Q - Qg), u1, u2)

    # Implicit system dynamics
    f_impl = xdot - f_expl

    # Populate model object
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Names for labelling plots
    model.t_label = "$t$ [s]"
    model.x_labels = ["$\\Omega (rad/s)$", "$\\theta (deg)$", "$Q_g (kNm)$"]
    model.u_labels = ["$\\dot{\\theta} (deg/s)$", "$\\dot{Q_g} (kNm)$"]

    return model
