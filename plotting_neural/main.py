import numpy as np
import pandas as pd

from parameters import Jonkman
from utils import plot_robot

params = Jonkman()
wind = params.wind_speed

# Load your CSV
df = pd.read_csv("results_dump8ms.csv")

# Constants exactly from your closed_loop_simulation
c1 = 0.5176
c2 = 116
c3 = 0.4
c4 = 5
c5 = 21
c6 = 0.0068

# Extract sorted unique time points
time_points = np.sort(df['t'].unique())

# States and inputs labels (your measurements exactly)
state_labels = [r"$\Omega$", r"$\theta$", r"$Q_g$"]
input_labels = [r"$\dot{\theta}$", r"$\dot{Q}_g$"]

# Prepare arrays to hold states and inputs
X_traj = np.zeros((len(time_points), len(state_labels)))
U_traj = np.zeros((len(time_points) - 1, len(input_labels)))  # inputs one less than states

for i, t in enumerate(time_points):
    df_t = df[df['t'] == t]
    # Get states at time t
    for j, state in enumerate(state_labels):
        val = df_t.loc[df_t['measurement'] == state, 'value']
        X_traj[i, j] = val.values[0] if not val.empty else np.nan
    # Get inputs at time t only if i < len(time_points)-1
    if i < len(time_points) - 1:
        for k, u in enumerate(input_labels):
            val = df_t.loc[df_t['measurement'] == u, 'value']
            U_traj[i, k] = val.values[0] if not val.empty else np.nan

# Compute power series exactly like your closed_loop_simulation
Pwr_series = np.zeros(len(time_points))
for i in range(len(time_points)):
    Omega = X_traj[i, 0]
    theta = X_traj[i, 1]
    L = Omega * params.radius / params.wind_speed

    denom = (1 / (L + 0.08 * theta) - 0.035 / (theta**3 + 1))
    if denom == 0:
        Li = 1e-6
    else:
        Li = 1 / denom

    Cp1 = c1 * (c2 / Li - c3 * theta - c4)
    Cp2 = np.exp(-c5 / Li)
    Cp3 = c6 * L
    Cp = Cp1 * Cp2 + Cp3

    Pout = (0.5 * params.air_density * np.pi * (params.radius ** 2) * (params.wind_speed ** 3) * Cp) / 1000
    Pwr_series[i] = Pout

print(Pout)

# Now plot
plot_robot(
    Pwr_series,
    wind,
    goal=wind*7.0/61.5,          # replace if you want a goal
    shooting_nodes=time_points,
    u_max=[None, None], # or set your limits
    U=U_traj,
    X_traj=X_traj,
    x_labels=state_labels,
    u_labels=input_labels,
    time_label="$t$",
)
