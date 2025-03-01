from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from robot_model import export_robot_model
import numpy as np
import scipy.linalg
from utils import plot_robot

# find optimal Omega
from parameters import Jonkman
params = Jonkman()
wind_speed = params.wind_speed
radius = params.radius
omega_reference = 7*wind_speed/radius

'''
    Sim is just one horizon
    x0 must be a member of the feasible set XF
    Any point in XF must be able to reach xN in one horizon

'''

x0 = np.array([0.5, 1e-3, 1e-3])

yref = np.array([omega_reference, 0.0, 0.0, 0.0, 0.0])
yref_N = np.array([omega_reference, 0.0, 0.0])   

# set cost
Q_mat = 1 * np.diag([100, 0, 0])
R_mat = 1 * np.diag([1e-2, 1e-2])


# simulation time
Ts = 1.0                    # Sample time
N_horizon = 50              # number of steps in horizon
time_of_sim = Ts*N_horizon  # Length of simulation horizon

# state constraints
max_Omega   = 1.267     # rad/s
max_theta   = 1.5708    # rad
max_Qg      = 47402.91  # N*m

# input constraints
max_pitch_rate  = 0.139626  # rad/s
max_torque_rate = 15000.0   # N*m/s

def create_ocp_solver_description() -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # # set state constraints
    # ocp.constraints.lbx = np.array([-max_Omega, -max_theta, -max_Qg])
    # ocp.constraints.ubx = np.array([+max_Omega, +max_theta, +max_Qg])
    # ocp.constraints.idxbx = np.array([0, 1, 2])

    # # set input constraints
    # ocp.constraints.lbu = np.array([-max_pitch_rate, -max_torque_rate])
    # ocp.constraints.ubu = np.array([max_pitch_rate, max_torque_rate])
    # ocp.constraints.idxbu = np.array([0, 1])

    # set initial condition
    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON" 
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # set prediction horizon
    ocp.solver_options.tf = time_of_sim

    return ocp


def closed_loop_simulation():

    # create solvers
    ocp = create_ocp_solver_description()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = x0
    simX[0, :] = xcurrent

    # closed loop
    for i in range(Nsim):
        # update yref
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", yref)
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        status = acados_ocp_solver.get_status()

        if status not in [0, 2]:
            acados_ocp_solver.print_statistics()
            plot_robot(
                np.linspace(0, time_of_sim / N_horizon * i, i + 1),
                simU[:i, :],
                simX[: i + 1, :],
            )
            raise Exception(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )

        # simulate system
        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

    # plot results
    plot_robot(
        np.linspace(0, time_of_sim / N_horizon * Nsim, Nsim + 1), [None, None],  simU, simX,
        x_labels=model.x_labels, u_labels=model.u_labels, time_label=model.t_label
    )


if __name__ == "__main__":
    closed_loop_simulation()