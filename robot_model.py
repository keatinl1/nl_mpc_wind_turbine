from acados_template import AcadosModel
from casadi import SX, vertcat, pi, exp

from parameters import Jonkman
params = Jonkman()

# Reference for model equations:
# https://backend.orbit.dtu.dk/ws/portalfiles/portal/5832126/prod21318234687066.2070.pdf

def export_robot_model() -> AcadosModel:
    model_name = "turbine"
    
    # Define states (x3)
    Omega = SX.sym('Omega')  # Rotor speed
    theta = SX.sym('theta')  # Blade pitch angle
    Qg = SX.sym('Qg')  # Generator torque
    x = vertcat(Omega, theta, Qg)
    
    # Define inputs (x2)
    u1 = SX.sym('u1')  # Pitch rate
    u2 = SX.sym('u2')  # Generator torque rate
    u = vertcat(u1, u2)

    # Define dynamics states (x3)
    Omega_dot = SX.sym("Omega_dot")
    theta_dot = SX.sym("theta_dot")
    Qg_dot = SX.sym("Qg_dot")
    xdot = vertcat(Omega_dot, theta_dot, Qg_dot)

    # Define/import some constants
    rho = 1.225
    C1, C2, C3, C4, C5, C6 = 0.5176, 116, 0.4, 5, 21, 0.0068
    V   = params.wind_speed
    R   = params.radius
    Jt  = params.moment_o_inertia
    
    # Calc Q which is used in Omega_dot
    lambda_i = 1/(1/(((R * Omega) / V) + 0.08*theta) - 0.035/(theta**3 + 1))
    Cp = C1*((C2/lambda_i) - C3*theta - C4)*exp(-C5/lambda_i) + C6*((R * Omega) / V)
    Q = (0.5*rho*pi*(R**2)*(V**3)*Cp)/Omega

    # Explicit system dynamics
    f_expl = vertcat((1/Jt)*(Q - Qg), 
                     u1, 
                     u2)

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
    model.x_labels = ["$\\Omega$", "$\\theta$", "$Q_g$"]
    model.u_labels = ["$\\dot{\\theta}$", "$\\dot{Q_g}$"]

    return model
