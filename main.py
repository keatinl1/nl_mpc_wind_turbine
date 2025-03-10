from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from robot_model import export_robot_model
import numpy as np
import scipy.linalg
from utils import plot_robot

# find the reference for turbine speed, 
# Tip-speed ratio (lambda) is approx 7 at peak power
from parameters import Jonkman
param = Jonkman()
wind = param.wind_speed
ref_Omega = 7*wind / 61.5 

'''
    sim is 2000 time steps long
    horizon is 50 time steps
    each time step a sequence of 50 inputs (u) is generated, 
    the first u is applied to the system,
    then another sequence is generated at the next time step .

    x0 is the initial state of horizon,
    xN is the terminal state of horizon.

    Xf is the feasible set, 
    set of initial states which there exists a feasible control sequence that drives the system to the terminal set.
    x0 must be a member of the feasible set Xf

    Xt is the terminal set, 
    the terminal set is invariant (once inside there exist inputs to stay there), 
        all constraints (x, u) are satified inside the set (considered inactive)
        and the terminal cost is a continuous Lyapunov function (x_k+1 < x_k) inside the terminal set
    xN must be a member of the terminal set Xt

'''

x0 = np.array([1e-6, 1e-6, 1e-6])

yref = np.array([ref_Omega, 0.0, 0.0, 0.0, 0.0])
yref_N = np.array([ref_Omega, 0.0, 0.0])   

# set cost
Q_mat = 1 * np.diag([6, 0, 0])      # only care about Omega, as long as other states are within constraints
R_mat = 1 * np.diag([900, 1e-6])    # use more torque than blade pitch to achieve goal, i.e. gen more power

# Q_mat = np.diag([1, 0, 0])      #Omega, theta, Qg
# R_mat = np.diag([1, 1e-3])    #theta_dot, Qg_dot

# simulation time
Ts = 1.0                    # Sample time (s)
N_horizon = 50              # number of steps in horizon
time_of_sim = Ts*N_horizon  # Length of simulation horizon in seconds

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

    # LLS cost is functionally equivalent to a quadratic cost
    # see: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    # eg:
    # where,    x e R^n, 
    #           Q e R^{n x n} and is diagonal.
    #           (x-xref)*Q*(x-xref) == sum((x-xref)^2*diag(Q))
    # (_e means at the 'end' or terminal of horizon)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Number of references 
    # the state x_k is dependant on x_k-1 and u_k-1, NOT u_k, hence why u_N is not considered in the optimization
    ny = nx + nu
    ny_e = nx

    # weights
    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    # ive kept the Vx and Vu matrices as identity matrices
    # im already ignoring states and inputs that are not relevant
    # in Q and R so its a bit redundant to zero them here also 
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set state constraints
    ocp.constraints.lbx = np.array([-max_Omega, 0, -max_Qg])
    ocp.constraints.ubx = np.array([+max_Omega, +max_theta, +max_Qg])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # set input constraints
    ocp.constraints.lbu = np.array([-max_pitch_rate, -max_torque_rate])
    ocp.constraints.ubu = np.array([max_pitch_rate, max_torque_rate])
    ocp.constraints.idxbu = np.array([0, 1])

    # set initial condition
    ocp.constraints.x0 = x0

    # set solver options, just the default options really
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON" 
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    # problem was getting stuck so i switched to rti
    # ocp.solver_options.nlp_solver_type = "SQP_RTI"

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
    Nsim = 2000 # how many timesteps the sim should run for
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

    print(xcurrent)

    # plot results
    plot_robot(
        wind, ref_Omega, np.linspace(0, time_of_sim / N_horizon * Nsim, Nsim + 1), [None, None],  simU, simX,
        x_labels=model.x_labels, u_labels=model.u_labels, time_label=model.t_label
    )

if __name__ == "__main__":
    closed_loop_simulation()
