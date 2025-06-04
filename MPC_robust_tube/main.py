'''
Robust tube-based NMPC for wind turbines 

Bounded disturbances:
- Omega ±0.4
- Theta ±1.0
- Qg    ±0.47

author: Luke Keating
date: 03/06/2025

'''

# === Standard imports ===
import scipy.linalg
import numpy as np
import control

# === Project modules ===
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from src.turbine_model import export_robot_model
from src.set.set_load_new import ZfSet, ZSet
from src.linear_model import LinearModel
from src.parameters import Jonkman
from utils import plot_robot

# === Instantiate Classes ===
linear_model = LinearModel()
params = Jonkman()
terminal_set = ZfSet()
stage_set = ZSet()

# === Time settings ===
ts = 0.05
N_horizon = 200
T_horizon = N_horizon * ts

# === References and initial states ===
Omega_ref = min(1.267, round(params.wind_speed*7.0 / params.radius, 3))
# adjusted starting point to be well within the robust set
Z0 = np.array([0.1, 2.0, 0.0])
X0 = np.array([0.1, 2.0, 0.0])

prev_disturbance = np.zeros(3)

Q = np.diag([100.0, 1e-6, 1e-2])
R = np.diag([10.0, 1e-3])

def create_nominal_z_ocp() -> AcadosOcp:
    # === Create OCP object and configure ===
    ocp = AcadosOcp()
    model = export_robot_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows() 
    ny = nx + nu
    ny_e = nx

    # === Horizon === 
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon
    
    # === Cost weights ===
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = control.dlqr(linear_model.A, linear_model.B, Q, R)[1]

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # === Cost mapping ===
    ocp.cost.Vx = np.vstack([
        np.eye(nx),
        np.zeros((nu, nx))
    ])
    ocp.cost.Vu = np.vstack([
        np.zeros((nx, nu)),
        np.eye(nu)
    ])
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # === Constraints: State ===
    ocp.constraints.C = stage_set.A
    ocp.constraints.lg = -1e10 * np.ones_like(stage_set.b)
    ocp.constraints.ug = stage_set.b.transpose()
    ocp.constraints.D = np.zeros((stage_set.A.shape[0], nu)) # D is necessary for stage

    # === Constraints: Input ===
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = -np.array([params.max_pitch_rate, params.max_torque_rate])
    ocp.constraints.ubu =  np.array([params.max_pitch_rate, params.max_torque_rate])

    # === Constraints: Terminal state ===
    ocp.constraints.C_e = terminal_set.A
    ocp.constraints.lg_e = -1e10 * np.ones(terminal_set.A.shape[0])
    ocp.constraints.ug_e = terminal_set.b.reshape(-1)

    # === Further options ===
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # === Initial state ===
    ocp.constraints.x0 = Z0

    return ocp

def create_actual_x_ocp() -> AcadosOcp:
    # === Create OCP object and configure ===
    ocp = AcadosOcp()
    model = export_robot_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows() 
    ny = nx + nu
    ny_e = nx

    # === Horizon === 
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    # Q_s = np.diag([10.0, 1e-6, 1e-3])
    # R_s = np.diag([1.0, 1e-6])

    # === Cost weights ===
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = np.zeros((nx, nx))

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # === Cost mapping ===
    ocp.cost.Vx = np.vstack([
        np.eye(nx),
        np.zeros((nu, nx))
    ])
    ocp.cost.Vu = np.vstack([
        np.zeros((nx, nu)),
        np.eye(nu)
    ])
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # === Constraints: State ===
    ocp.constraints.lbx = np.array([1e-6, 0.0, -params.max_Qg])
    ocp.constraints.ubx = np.array([+params.max_Omega, +params.max_theta, +params.max_Qg])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # === Constraints: Input ===
    ocp.constraints.lbu = np.array([-params.max_pitch_rate, -params.max_torque_rate])
    ocp.constraints.ubu = np.array([params.max_pitch_rate, params.max_torque_rate])
    ocp.constraints.idxbu = np.array([0, 1])

    # === Constraints: Terminal state ===
    # initialise as this, but we will update in the loop
    ocp.constraints.lbx_e = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubx_e = np.array([0.0, 0.0, 0.0])
    ocp.constraints.idxbx_e = np.array([0, 1, 2])

    # === Further options ===
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # === Initial state ===
    ocp.constraints.x0 = X0

    return ocp

def get_disturbance():
    d0 = np.random.uniform(-0.01, 0.01)    # Omega
    d1 = np.random.uniform(-1.0, 1.0)    # Theta
    d2 = np.random.uniform(-0.47, 0.47)  # Qg
    return np.array([d0, d1, d2])

def closed_loop_simulation():

    # create nominal sys
    ocp_z = create_nominal_z_ocp()
    model_z = ocp_z.model
    acados_ocp_solver_z = AcadosOcpSolver(ocp_z)
    acados_integrator_z = AcadosSimSolver(ocp_z)

    # create actual sys
    ocp_x = create_actual_x_ocp()
    acados_ocp_solver_x = AcadosOcpSolver(ocp_x)
    acados_integrator_x = AcadosSimSolver(ocp_x)

    # prep sim
    Nsim = 2000
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
    yref = np.array([Omega_ref, 0, params.max_Qg, 0, 0])
    yref_N = np.array([Omega_ref, 0, params.max_Qg])
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
        
        # First-order low-pass filter on w
        alpha = 0.05  # LPF smoothing factor, 0 < alpha <= 1
        if i == 0:
            w = get_disturbance()
        else:
            w_raw = get_disturbance()
            w = alpha * w_raw + (1 - alpha) * w_prev
        w_prev = w

        simX[i + 1, :] = xcurrent + w

        if i % 100 == 0:
            print(round(i*100/Nsim, 2), "% complete")

    print("100.0 % complete\n")
    print("Reference: ", Omega_ref)
    print("Achieved:  ", round(xcurrent[0], 4), "\n")

    print("Final state: ", xcurrent)

    plot_robot(
        params.wind_speed, Omega_ref,
        np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1),
        [None, None],  # u_max if needed
        simU,
        simX,
        Z_traj=simZ,
        x_labels=model_z.x_labels,
        u_labels=model_z.u_labels,
        time_label=model_z.t_label
    )

if __name__ == "__main__":
    closed_loop_simulation()
