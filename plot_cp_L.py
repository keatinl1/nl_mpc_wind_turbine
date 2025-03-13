import numpy as np
import matplotlib.pyplot as plt

# Constants
c1 = 0.5176
c2 = 116
c3 = 0.4  # Keep c3 as is if it was for degrees, we scale theta directly
c4 = 5
c5 = 21
c6 = 0.0068

# Define L values (avoid division by zero)
L = np.arange(0.1, 14.1, 0.1)  # Start from 0.1 to prevent 1/0 error

# Theta values in degrees (scaled for radians)
theta_values = np.array([0, 5, 10, 12.5, 15, 20, 30])

plt.figure(figsize=(8, 5))

# Loop through theta values directly in radians
for theta in theta_values:

    Li = 1 / (1 / (L + 0.08 * theta) - 0.035 / (theta**3 + 1))
    Cp1 = c1 * (c2 / Li - c3 * theta - c4)  # Using scaled theta here
    Cp2 = np.exp(-c5 / Li)
    Cp3 = c6 * L
    Cp = Cp1 * Cp2 + Cp3
    plt.plot(L, Cp, label=f'θ = {theta:.2f} rad')  # Display radians in the legend

# Formatting the plot
plt.xlabel('$\lambda$')
plt.ylabel('$C_p$')
plt.title('Power Coefficient vs Tip Speed Ratio')
plt.xlim([0, 14])  # Define x-axis range
plt.ylim([0, 0.5])  # Define y-axis range
plt.legend()
plt.grid()
plt.show()
