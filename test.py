import numpy as np

Jt = 1.0
rho = 1.225
R = 63.0
V = 5.0
Omega = 35.0
theta = 0.1
Qg = 2.0

# # Cp 
# C1, C2, C3, C4, C5, C6 = 0.5176, 116, 0.4, 5, 21, 0.0068
# lambda_ = (R * Omega) / V

# lambda_i_pre = 1 / (1 / (lambda_ + 0.08 * theta)) - (0.035 / (theta**3 + 1))
# lambda_i = 1/lambda_i_pre

# Cp = (C1 * ((C2 / lambda_i) - C3 * theta - C4) * np.exp(-C5 / lambda_i)) + (C6 * lambda_)

# # dynamics
# Omega_dot = (1 / Jt) * ((0.5 * rho * np.pi * R**2 * V**3 * Cp )/ Omega - Qg)
# print(Omega_dot)


# Cp 
C1, C2, C3, C4, C5, C6 = 0.5176, 116, 0.4, 5, 21, 0.0068

lambda_ = (R * Omega) / V
lambda_i = lambda_ + 0.08*theta - (theta**3 + 1)/0.035

Cp = C1*((C2/lambda_i) - C3*theta - C4)*np.exp(-C5/lambda_i) + C6*lambda_

Q = (0.5*rho*np.pi*(R**2)*(V**3)*Cp)/Omega

print((1/Jt)*(Q - Qg))
