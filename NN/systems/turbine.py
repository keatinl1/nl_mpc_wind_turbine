"""Define a dymamical system for a wind turbine"""
from typing import Tuple, Optional, List

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class Turbine(ControlAffineSystem):
    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 2

    # State indices
    OMEGA = 0
    THETA = 1
    QG    = 2

    # Control indices
    U1 = 0 # theta dot
    U2 = 1 # Qg dot

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.05,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        valid = True

        # Make sure all needed parameters were provided
        valid = valid and "R" in params
        valid = valid and "I" in params
        valid = valid and "p" in params
        valid = valid and "V" in params

        # Make sure all parameters are physically valid
        valid = valid and params["R"] > 0
        valid = valid and params["I"] > 0
        valid = valid and params["p"] > 0
        valid = valid and params["V"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Turbine.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Turbine.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Turbine.OMEGA] = 1.2670
        upper_limit[Turbine.THETA] = 90.0
        upper_limit[Turbine.QG]    = 47.40291

        lower_limit = torch.ones(self.n_dims)
        lower_limit[Turbine.OMEGA] = 0.0
        lower_limit[Turbine.THETA] = 0.0
        lower_limit[Turbine.QG]    = -47.40291

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # define upper and lower limits based around the nominal equilibrium input

        upper_limit = torch.ones(self.n_controls)
        upper_limit[Turbine.U1] = 8.0
        upper_limit[Turbine.U2] = 15.0

        lower_limit = torch.ones(self.n_controls)
        lower_limit[Turbine.U1] = -8.0
        lower_limit[Turbine.U2] = -15.0

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        O_l = 1e-6
        O_u = 1.27

        t_l = 0.0
        t_u = 90.0

        qg_l = -47.0
        qg_u = 47.0

        omega_safe = (x[:, Turbine.OMEGA] >= O_l) & (x[:, Turbine.OMEGA] <= O_u)
        theta_safe = (x[:, Turbine.THETA] >= t_l) & (x[:, Turbine.THETA] <= t_u)
        qg_safe = (x[:, Turbine.QG] >= qg_l) & (x[:, Turbine.QG] <= qg_u)

        safe_mask = omega_safe & theta_safe & qg_safe

        return safe_mask
    
    def unsafe_mask(self, x):
        O_l = 1e-6
        O_u = 1.27

        t_l = 0.0
        t_u = 90.0

        qg_l = -47.0
        qg_u = 47.0

        omega_unsafe = (x[:, Turbine.OMEGA] <= O_l) | (x[:, Turbine.OMEGA] >= O_u)
        theta_unsafe = (x[:, Turbine.THETA] <= t_l) | (x[:, Turbine.THETA] >= t_u)
        qg_unsafe = (x[:, Turbine.QG] <= qg_l) | (x[:, Turbine.QG] >= qg_u)

        unsafe_mask = omega_unsafe | theta_unsafe | qg_unsafe

        return unsafe_mask    

    def goal_mask(self, x):
        omega_near_goal = (x[:, Turbine.OMEGA] - 0.626).abs() <= 0.3
        qg_near_goal = (x[:, Turbine.QG]).abs() <= 47.0       # Example
        return omega_near_goal & qg_near_goal

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        R, I, p, V = params['R'], params['I'], params['p'], params['V']
        # and state variables
        Omega = x[:, Turbine.OMEGA]
        theta = x[:, Turbine.THETA]
        Qg    = x[:, Turbine.QG]

        epsilon = 1e-6
        Omega_clamped = x[:, Turbine.OMEGA].clamp(min=epsilon)
        theta_clamped = x[:, Turbine.THETA].clamp(min=epsilon)

        # The derivatives of theta is just its velocity
        c1 = 0.5176
        c2 = 116
        c3 = 0.4
        c4 = 5
        c5 = 21
        c6 = 0.0068

        L = Omega_clamped * R / V
        Li = 1 / (1 / (L + 0.08 * theta_clamped) - 0.035 / (theta_clamped**3 + 1))
        Cp1 = c1 * (c2 / Li - c3 * theta_clamped - c4)  # Using scaled theta here
        Cp2 = torch.exp(-c5 / Li)
        Cp3 = c6 * L
        Cp = Cp1 * Cp2 + Cp3

        pi = 22/7

        Q = (0.5*p*pi*(R**2)*(V**3)*Cp)/(Omega_clamped*1000)

        # Explicit system dynamics
        f[:, Turbine.OMEGA, 0] = (1000/I)*(Q - Qg*97)
        f[:, Turbine.THETA, 0] = 0
        f[:, Turbine.QG, 0]    = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Effect on theta dot
        g[:, Turbine.THETA, Turbine.U1] = 1.0
        g[:, Turbine.QG, Turbine.U2]    = 1.0

        return g
