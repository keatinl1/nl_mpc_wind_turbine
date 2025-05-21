import os
import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

# def plot_robot(
#     wind_speed,
#     goal,
#     shooting_nodes,
#     u_max,
#     U,
#     X_traj,
#     x_labels,
#     u_labels,
#     time_label="$t$",
#     latexify=True,
#     plt_show=True,
# ):
#     """
#     Params:
#         shooting_nodes: time values of the discretization
#         u_max: maximum absolute value of u
#         U: array with shape (N_sim-1, nu) or (N_sim, nu)
#         X_traj: array with shape (N_sim, nx)
#         latexify: latex style plots
#     """

#     if latexify:
#         latexify_plot()

#     N_sim = X_traj.shape[0]
#     nx = X_traj.shape[1]
#     nu = U.shape[1]
#     fig, axs = plt.subplots(nx + nu, 1, figsize=(9, 9), sharex=True)  # Add one more subplot for the power series

#     t = shooting_nodes

#     for i in range(nx):
#         plt.subplot(nx + nu, 1, i + 1)  # Adjust index to start from 1
#         (line,) = plt.plot(t, X_traj[:, i])

#         plt.ylabel(x_labels[i])
#         plt.grid()

#         # Plot goal as a horizontal dashed line for the first state
#         if i == 0:
#             plt.title(f"Wind speed: {wind_speed} m/s")
#             plt.axhline(y=goal, color="r", linestyle="--", label=f"Goal, {goal:.2f} rad/s")
#             plt.legend()

#     # Plot controls after states
#     for i in range(nu):
#         plt.subplot(nx + nu, 1, nx + i + 1)  # Controls start after states
#         (line,) = plt.step(t, np.append([U[0, i]], U[:, i]), color="tab:orange")

#         plt.ylabel(u_labels[i])
#         # plt.xlabel(time_label)
#         if u_max[i] is not None:
#             plt.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
#             plt.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
#             plt.ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
#         plt.grid()


#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

#     axs[0].set_xlim([t[0], t[-1]])

#     if plt_show:
#         plt.show()

def plot_robot(
    wind_speed,
    goal,
    shooting_nodes,
    u_max,
    U,
    X_traj,
    Z_traj=None,
    x_labels=None,
    u_labels=None,
    time_label="$t$",
    latexify=True,
    plt_show=True,
):
    if latexify:
        latexify_plot()

    N_sim = X_traj.shape[0]
    nx = X_traj.shape[1]
    nu = U.shape[1]
    fig, axs = plt.subplots(nx + nu, 1, figsize=(10, 10), sharex=True)

    t = shooting_nodes

    for i in range(nx):
        plt.subplot(nx + nu, 1, i + 1)
        plt.plot(t, X_traj[:, i], label="Actual X", color="tab:blue")
        if Z_traj is not None:
            plt.plot(t, Z_traj[:, i], label="Nominal Z", color="tab:green", linestyle="--")

        plt.ylabel(x_labels[i])
        plt.grid()

        if i == 0:
            plt.title(f"Wind speed: {wind_speed} m/s")
            plt.axhline(y=goal, color="r", linestyle="--", label=f"Goal: {goal:.2f} rad/s")
            plt.legend()

    for i in range(nu):
        plt.subplot(nx + nu, 1, nx + i + 1)
        plt.step(t, np.append([U[0, i]], U[:, i]), color="tab:orange", label=u_labels[i])
        plt.ylabel(u_labels[i])
        if u_max[i] is not None:
            plt.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
        plt.grid()

    axs[0].set_xlim([t[0], t[-1]])
    plt.subplots_adjust(hspace=0.4)

    if plt_show:
        plt.show()
