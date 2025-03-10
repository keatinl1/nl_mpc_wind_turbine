import numpy as np
import matplotlib.pyplot as plt
from parameters import Jonkman
from robot_model import export_robot_model
from casadi import SX, pi, exp

# Initialize parameters
params = Jonkman()

# Define the wind turbine model using the export_robot_model function
model = export_robot_model()

# System Dynamics
def system_dynamics(x, u):
    Omega, theta, Qg = x
    u1, u2 = u

    # Constants
    rho = 1.225
    C1, C2, C3, C4, C5, C6 = 0.5176, 116, 0.4, 5, 21, 0.0068
    V = params.wind_speed
    R = params.radius
    Jt = params.moment_o_inertia

    # Calculate lambda_i and Cp (power coefficient)
    lambda_i = 1 / (1 / (((R * Omega) / V) + 0.08 * theta) - 0.035 / (theta**3 + 1))
    Cp = C1 * ((C2 / lambda_i) - C3 * theta - C4) * exp(-C5 / lambda_i) + C6 * ((R * Omega) / V)

    # Compute Q (Aerodynamic Torque)
    if Omega > 0:  # Avoid division by zero
        Q = (0.5 * rho * pi * (R**2) * (V**3) * Cp) / Omega
    else:
        Q = 0  # If Omega is zero, no aerodynamic torque

    # State derivatives
    Omega_dot = (Q - Qg) / Jt
    theta_dot = u1
    Qg_dot = u2

    return np.array([Omega_dot, theta_dot, Qg_dot])

# Open-loop Simulation Function
def open_loop_simulate(x0, u_trajectory, T, dt):
    N = int(T / dt)
    time = np.linspace(0, T, N)
    
    x = np.array(x0)
    x_history = np.zeros((N, 3))
    u_history = np.zeros((N, 2))

    for i in range(N):
        x_history[i, :] = x
        u_history[i, :] = u_trajectory[i, :]
        
        dx = system_dynamics(x, u_trajectory[i, :])
        x = x + dx * dt  # Euler integration step

    return time, x_history, u_history

# Initial Conditions
x0 = [1e-3, 0.0, 0.0]  # Initial [Omega, theta, Qg]

# Simulation Parameters
T = 1000
dt = 1  
N = int(T / dt)

# Define Control Inputs (Set to Zero for Open-Loop)
u_trajectory = np.zeros((N, 2))
u_trajectory[0, :] = 1.0  # very first timestep is 1 to imitate an impulse

# Run Open-Loop Simulation
time, x_history, u_history = open_loop_simulate(x0, u_trajectory, T, dt)

# Updated Plotting Function
def plot_robot(wind_speed, shooting_nodes, u_max, U, X_traj, x_labels, u_labels, time_label="$t$", latexify=False, plt_show=True):
    
    N_sim, nx = X_traj.shape
    
    nu = U.shape[1]

    fig, axs = plt.subplots(nx + nu + 1, 1, figsize=(9, 9), sharex=True)
    t = shooting_nodes

    # Plot states
    for i in range(nx):
        axs[i].plot(t, X_traj[:, i])
        axs[i].set_ylabel(x_labels[i])
        axs[i].grid()

        if i == 0:
            axs[i].set_title(f"Wind speed: {wind_speed} m/s")

    # Plot control inputs
    for i in range(nu):
        axs[nx + i].step(t, U[:, i], color="tab:orange")
        axs[nx + i].set_ylabel(u_labels[i])
        axs[nx + i].grid()
        if u_max[i] is not None:
            axs[nx + i].hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            axs[nx + i].hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)

    # Power Output (Omega * Qg * Constant)
    axs[nx + nu].plot(t, X_traj[:, 0] * X_traj[:, -1] * 97 * 0.944, color="tab:red")
    axs[nx + nu].set_ylabel("$P_e$")
    axs[nx + nu].set_xlabel(time_label)
    axs[nx + nu].grid()

    plt.subplots_adjust(hspace=0.4)
    axs[0].set_xlim([t[0], t[-1]])

    if plt_show:
        plt.show()


print(x_history[-1, :])

# Call the Plot Function
plot_robot(
    wind_speed=params.wind_speed,
    shooting_nodes=time,
    u_max=[None, None],  # No input limits
    U=u_history,
    X_traj=x_history,
    x_labels=["$\\Omega$", "$\\theta$", "$Q_g$"],
    u_labels=["$\\dot{\\theta}$", "$\\dot{Q_g}$"],
)
