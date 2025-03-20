import scipy.linalg
import numpy as np

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from parameters import Jonkman
from turbine_model import export_robot_model
from utils import plot_robot


params = Jonkman()
wind = params.wind_speed
Omega_ref = round(wind*7 / params.radius, 4)

# N_horizon = 50  # Define the number of discretization steps
N_horizon = 20
ts = 0.05
T_horizon = N_horizon * ts  # Define the horizon time

X0 = np.array([1e-6, 1e-6, 1e-6])  # Intital state , avoid division by zero

# constraints
max_Omega = params.max_Omega
max_theta = params.max_theta
max_Qg    = params.max_Qg
max_pitch_rate  = params.max_pitch_rate
max_torque_rate = params.max_torque_rate

Pwr_series = []

def create_ocp_solver_description() -> AcadosOcp:

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    # working for 11m/s
    # Q_mat = np.diag([10, 0.1, 0])
    # R_mat = np.diag([1000, 1e-4])

    # reaches both
    # Q_mat = np.diag([10, 0, 0])
    # R_mat = np.diag([1, 1])

    # Q_mat = np.diag([10, 1e-6, 1e-6])
    # R_mat = np.diag([1e-2, 1e-3])

    Q_mat = np.diag([10, 1e-6, 1e-6])
    R_mat = np.diag([1e-2, 1e-6])

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
    ocp.constraints.lbx = np.array([-max_Omega, 0, 0])
    ocp.constraints.ubx = np.array([+max_Omega, +max_theta, +max_Qg])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # set input constraints
    ocp.constraints.lbu = np.array([-max_pitch_rate, -max_torque_rate])
    ocp.constraints.ubu = np.array([max_pitch_rate, max_torque_rate])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

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
    Nsim = 10000
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = X0
    simX[0, :] = xcurrent

    yref = np.array([Omega_ref, 0, max_Qg, 0, 0])
    yref_N = np.array([Omega_ref, 0, max_Qg])

    c1 = 0.5176
    c2 = 116
    c3 = 0.4
    c4 = 5
    c5 = 21
    c6 = 0.0068

    print("\n")

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
                None,
                simU[:i, :],
                simX[: i + 1, :],
            )
            raise Exception(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )
        

        L = xcurrent[0] * params.radius / params.wind_speed
        Li = 1 / (1 / (L + 0.08 * xcurrent[1]) - 0.035 / (xcurrent[1]**3 + 1))
        Cp1 = c1 * (c2 / Li - c3 * xcurrent[1] - c4)  # Using scaled theta here
        Cp2 = np.exp(-c5 / Li)
        Cp3 = c6 * L
        Cp = Cp1 * Cp2 + Cp3

        Pout = (0.5*params.air_density*np.pi*(params.radius**2)*(params.wind_speed**3)*Cp)/1000
        Pwr_series.append(Pout)

        # simulate system
        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

        if i % 100 == 0:
            print(i*100/Nsim, "% complete")

    print("100.0 % complete\n")
    print("Reference: ", Omega_ref)
    print("Achieved:  ", round(xcurrent[0], 4), "\n")

    print("Final state: ", xcurrent, "\n\nFinal power output: ", round(Pout, 2), "kW")

    plot_robot(
        Pwr_series, wind, Omega_ref, np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [None, None],  simU, simX,
        x_labels=model.x_labels, u_labels=model.u_labels, time_label=model.t_label
    )


if __name__ == "__main__":
    closed_loop_simulation()
