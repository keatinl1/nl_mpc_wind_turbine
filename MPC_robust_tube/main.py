# import libraries
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import numpy as np
import control

# import objects
from terminal_components import Terminal
from parameters import Jonkman
from linear_model import Lin_model

# import functions
from turbine_model import export_robot_model
from utils import plot_robot

# instantiate some objects
linear_model = Lin_model()
terminal = Terminal()
params = Jonkman()

# get data from classes
A, B = linear_model.A, linear_model.B
wind = params.wind_speed
max_Omega = params.max_Omega
max_theta = params.max_theta
max_Qg    = params.max_Qg
max_pitch_rate  = params.max_pitch_rate
max_torque_rate = params.max_torque_rate

# declare params
Omega_ref = min(1.267, round(wind*7.0 / params.radius, 3))
N_horizon = 75
ts = 0.05
T_horizon = N_horizon * ts

Z0 = np.array([1e-6, 1e-6, 1e-6])
X0 = np.array([1e-6, 1e-6, 1e-6])

def create_nominal_z_ocp() -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    ny = nx + nu
    ny_e = nx

    # Stage ==============================================================
    # Cost
    ocp.cost.cost_type = "LINEAR_LS"
    Q_mat = np.diag([10.0, 1e-3, 1e-6])
    R_mat = np.diag([10.0, 1e-6])
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    # Constraints
    # state
    ocp.constraints.lbx = np.array([1e-6, 0.0, -max_Qg])
    ocp.constraints.ubx = np.array([+max_Omega, +max_theta, +max_Qg])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # input
    ocp.constraints.lbu = np.array([-max_pitch_rate, -max_torque_rate])
    ocp.constraints.ubu = np.array([max_pitch_rate, max_torque_rate])
    ocp.constraints.idxbu = np.array([0, 1])

    # Terminal =========================================================
    # Cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    _, P, _ = control.dlqr(A, B, Q_mat, R_mat)
    ocp.cost.W_e = P
    ocp.cost.Vx_e = np.eye(nx)

    # set terminal state constraints
    ocp.constraints.C_e = terminal.A
    ocp.constraints.lg_e = -1e10 * np.ones_like(terminal.b)
    ocp.constraints.ug_e = terminal.b.transpose()

    # Output Cost (n/a) ===============================================
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    ocp.constraints.x0 = Z0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp

def create_actual_x_ocp() -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    ny = nx + nu
    ny_e = nx

    # Stage ==============================================================
    # Cost
    ocp.cost.cost_type = "LINEAR_LS"
    Q_mat = np.diag([10.0, 1e-3, 1e-6])
    R_mat = np.diag([10.0, 1e-6])
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    # Constraints
    # state
    ocp.constraints.lbx = np.array([1e-6, 0.0, -max_Qg])
    ocp.constraints.ubx = np.array([+max_Omega, +max_theta, +max_Qg])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # input
    ocp.constraints.lbu = np.array([-max_pitch_rate, -max_torque_rate])
    ocp.constraints.ubu = np.array([max_pitch_rate, max_torque_rate])
    ocp.constraints.idxbu = np.array([0, 1])

    # Terminal =========================================================
    # Cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e = np.zeros((nx, nx))
    ocp.cost.Vx_e = np.eye(nx)

    # Constraints
    # state
    ocp.constraints.lbx = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubx = np.array([0.0, 0.0, 0.0])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # Output Cost (doesnt exist) ========================================
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp

def get_disturbance(i, seed=42):
    np.random.seed(seed + i)  # deterministic

    # Relative scales for each state (say, 1% of range)
    delta_omega = 0.01 * (1.26 - 1e-6)
    delta_theta = 0.01 * (90.0 - 0.0)
    delta_qg    = 0.01 * (47.0 - (-47.0))

    # Sinusoidal + noise
    d0 = delta_omega * (np.sin(0.01 * i) + 0.2 * np.random.randn())
    d1 = delta_theta * (np.sin(0.0001 * i) + 0.2 * np.random.randn())
    d2 = delta_qg    * (np.sin(0.02 * i + 1.0) + 0.2 * np.random.randn())

    return np.array([abs(d0), abs(d1), abs(d2)])

def closed_loop_simulation():

    # create nominal sys
    ocp_z = create_nominal_z_ocp()
    model_z = ocp_z.model
    acados_ocp_solver_z = AcadosOcpSolver(ocp_z)
    acados_integrator_z = AcadosSimSolver(ocp_z)
    N_horizon = acados_ocp_solver_z.N

    # create actual sys
    ocp_x = create_actual_x_ocp()
    model_x = ocp_x.model
    acados_ocp_solver_x = AcadosOcpSolver(ocp_x)
    acados_integrator_x = AcadosSimSolver(ocp_x)
    N_horizon = acados_ocp_solver_x.N

    # prep sim
    Nsim = 2500
    nx = ocp_z.model.x.rows()
    nu = ocp_z.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    simZ = np.zeros((Nsim + 1, nx))
    simV = np.zeros((Nsim, nu))

    pred_Z = np.zeros((N_horizon, nx)) 

    # set initial conditions
    xcurrent = X0
    zcurrent = Z0

    simX[0, :] = xcurrent
    simZ[0, :] = zcurrent

    # reference is constaint for nominal
    yref = np.array([Omega_ref, 0, max_Qg, 0, 0])
    yref_N = np.array([Omega_ref, 0, max_Qg])
    for j in range(N_horizon):
        acados_ocp_solver_z.set(j, "yref", yref)
    acados_ocp_solver_z.set(N_horizon, "yref", yref_N)

    # closed loop
    for i in range(Nsim):

        # Z: NOMINAL 
        simV[i, :] = acados_ocp_solver_z.solve_for_x0(zcurrent)
        for k in range(N_horizon):
            pred_Z[k, :] = acados_ocp_solver_z.get(k, "x")
        z_status = acados_ocp_solver_z.get_status()

        if z_status not in [0, 2]:
            acados_ocp_solver_z.print_statistics()
            raise Exception(
                f"Z: returned status {z_status} in closed loop instance {i} with {xcurrent}"
            )

        # X: ACTUAL
        for j in range(N_horizon):
            acados_ocp_solver_x.set(j, "yref", np.concatenate((pred_Z[j, :], np.zeros(nu))))
        acados_ocp_solver_x.set(N_horizon, "yref", pred_Z[-1, :])    
        simU[i, :] = acados_ocp_solver_x.solve_for_x0(xcurrent)
        x_status = acados_ocp_solver_x.get_status()

        if x_status not in [0, 2]:
            acados_ocp_solver_x.print_statistics()
            raise Exception(
                f"X: returned status {x_status} in closed loop instance {i} with {xcurrent}"
            )


        # update system
        # z+ = f(z, v0)
        zcurrent = acados_integrator_z.simulate(zcurrent, simV[i, :])
        simZ[i + 1, :] = zcurrent

        # x+ = f(x, u0) + w
        xcurrent = acados_integrator_x.simulate(xcurrent, simU[i, :])
        w = get_disturbance(i)
        simX[i + 1, :] = xcurrent + w

        if i % 100 == 0:
            print(round(i*100/Nsim, 2), "% complete")

    print("100.0 % complete\n")
    print("Reference: ", Omega_ref)
    print("Achieved:  ", round(xcurrent[0], 4), "\n")

    print("Final state: ", xcurrent)

    plot_robot(
        wind, Omega_ref, np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [None, None],  simU, simX,
        x_labels=model_z.x_labels, u_labels=model_z.u_labels, time_label=model_z.t_label
    )

if __name__ == "__main__":
    closed_loop_simulation()
