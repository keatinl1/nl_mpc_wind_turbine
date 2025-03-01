import os
import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

def plot_robot(
    shooting_nodes,
    u_max,
    U,
    X_traj,
    x_labels,
    u_labels,
    time_label="$t$",
    latexify=True,
    plt_show=True,
):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_traj: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    N_sim = X_traj.shape[0]
    nx = X_traj.shape[1]
    nu = U.shape[1]
    fig, axs = plt.subplots(nx+nu+1, 1, figsize=(9, 9), sharex=True)

    t = shooting_nodes

    # Plot states first
    for i in range(nx):
        plt.subplot(nx + nu + 1, 1, i + 1)  # Adjust index to start from 1
        (line,) = plt.plot(t, X_traj[:, i])

        plt.ylabel(x_labels[i])
        # plt.xlabel(time_label)
        plt.grid()

    # Plot controls after states
    for i in range(nu):
        plt.subplot(nx + nu + 1, 1, nx + i + 1)  # Controls start after states
        (line,) = plt.step(t, np.append([U[0, i]], U[:, i]), color="tab:orange")

        plt.ylabel(u_labels[i])
        # plt.xlabel(time_label)
        if u_max[i] is not None:
            plt.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
        plt.grid()
    
    # Additional plot: first column vs. third column of X_traj by 95
    plt.subplot(nx + nu + 1, 1, nx + nu + 1)
    plt.plot(t, X_traj[:, 0]*X_traj[:, -1] * 97 * .944, color="tab:red")
    plt.ylabel("$P_e$")
    plt.xlabel("$t$ [s]")
    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    axs[0].set_xlim([t[0], t[-1]])

    if plt_show:
        plt.show()
