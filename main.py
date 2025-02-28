from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from robot_model import export_robot_model
import numpy as np
import scipy.linalg
from utils import plot_robot

X0 = np.array([1e-3, 0, 1e-3])  # Intital state, to avoid zero division make them very small
T_horizon = 50  # Length of simulation horizon

# state constraints
max_Omega   = 1.267 # rad/s
max_theta   = 1.5708 # rad
max_Qg      = 47402.91 # N*m
# input constraints
max_pitch_rate  = 0.139626 # rad/s
max_torque_rate = 15000.0 # N*m

def create_ocp_solver_description() -> AcadosOcp:
    N_horizon = 50  # Define the number of discretization steps

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    Q_mat = 1 * np.diag([0, 10, 0])
    R_mat = 1 * np.diag([1, 1])

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

    # set state constraints
    ocp.constraints.lbx = np.array([-max_Omega])
    ocp.constraints.ubx = np.array([+max_Omega])
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.lbx = np.array([-max_theta])
    ocp.constraints.ubx = np.array([+max_theta])
    ocp.constraints.idxbx = np.array([1])

    ocp.constraints.lbx = np.array([-max_Qg])
    ocp.constraints.ubx = np.array([+max_Qg])
    ocp.constraints.idxbx = np.array([2])

    # set input constraints
    ocp.constraints.lbu = np.array([-max_pitch_rate])
    ocp.constraints.ubu = np.array([max_pitch_rate])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbu = np.array([-max_torque_rate])
    ocp.constraints.ubu = np.array([max_torque_rate])
    ocp.constraints.idxbu = np.array([1])

    # set initial condition
    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON" 
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

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

    xcurrent = X0
    simX[0, :] = xcurrent

    yref = np.array([0.0, 0.50, 0.0, 0.0, 0.0])
    yref_N = np.array([0.0, 0.50, 0.0])

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
                np.linspace(0, T_horizon / N_horizon * i, i + 1),
                simU[:i, :],
                simX[: i + 1, :],
            )
            raise Exception(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )

        # simulate system
        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

    print("\nFinal system state:\n"
    "Omega (ref): {:.4f} ({})\n"
    "Theta (ref): {:.4f} ({})\n"
    "Qg    (ref): {:.4f} ({})".format(
        xcurrent[0], yref_N[0],
        xcurrent[1], yref_N[1],
        xcurrent[2], yref_N[2]
    ))

    # plot results
    plot_robot(
        np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [None, None],  simU, simX,
        x_labels=model.x_labels, u_labels=model.u_labels, time_label=model.t_label
    )


if __name__ == "__main__":
    closed_loop_simulation()