from acados_template import AcadosModel
from casadi import SX, vertcat, pi, exp

# Reference for model equations:
# https://backend.orbit.dtu.dk/ws/portalfiles/portal/5832126/prod21318234687066.2070.pdf

def export_robot_model() -> AcadosModel:
    model_name = "turbine"
    
    # Define state variables
    Omega = SX.sym('Omega')  # Rotor speed
    theta = SX.sym('theta')  # Blade pitch angle
    Qg = SX.sym('Qg')  # Generator torque
    x = vertcat(Omega, theta, Qg)
    
    # Control inputs
    u1 = SX.sym('u1')  # Pitch rate
    u2 = SX.sym('u2')  # Generator torque rate
    u = vertcat(u1, u2)

    # xdot
    Omega_dot = SX.sym("Omega_dot")
    theta_dot = SX.sym("theta_dot")
    Qg_dot = SX.sym("Qg_dot")

    xdot = vertcat(Omega_dot, theta_dot, Qg_dot)

    Jt = 1.0
    rho = 1.225
    R = 63.0
    V = 5.0
    
    # Cp 
    C1, C2, C3, C4, C5, C6 = 0.5176, 116, 0.4, 5, 21, 0.0068
    
    lambda_ = (R * Omega) / V
    inv_lambda_i = 1/(lambda_ + 0.08*theta) - 0.035/(theta**3 + 1)
    lambda_i = 1/inv_lambda_i
        
    Cp = C1*((C2/lambda_i) - C3*theta - C4)*exp(-C5/lambda_i) + C6*lambda_

    Q = (0.5*rho*pi*(R**2)*(V**3)*Cp)/Omega

    # dynamics
    f_expl = vertcat((1/Jt)*(Q - Qg), u1, u2)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$\\Omega$", "$\\theta$", "$Q_g$"]
    model.u_labels = ["$u_1 (\\dot{\\theta})$", "$u_2 (\\dot{Q_g})$"]

    return model
