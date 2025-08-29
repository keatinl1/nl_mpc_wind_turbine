## Nonlinear Model Predictive and Neural Control of Wind Turbines

This was the project I undertook for my master thesis at Institut polytechnique de Grenoble. 

The project was conducted in the department of mechanical, manufacturing and biomedical engineering at Trinity College Dublin under the supervision of Prof. Siyuan Zhan.

I explored the theory behind nominal and robust MPC deeply and learned much about optimisation in the process.

I also trained a neural network controller which had formal guarantees on safety and stability to compare. 

## System Model

The system model was taken from another master thesis [1] and has three states. 

Angular velocity ($\Omega$), blade pitch angle ($\theta$) and generator torque ($Q_g$). The two inputs were simply the blade pitch angle rate and the generator torque rate.

$$
\dot{\begin{pmatrix}
  \Omega \\
  \theta \\
  Q_g 
\end{pmatrix}}
= \begin{pmatrix}
  \frac{1}{J_t}\left(Q(V, \Omega, \theta) - Q_g\right) \\
  u_1 \\
  u_2
\end{pmatrix}
$$

## Methodology

For the MPC, I used the acados optimal control framework [2] and adopted one of the examples to fit my problem. 

For the set computations, I used the Multi-Parametric Toolbox [3].

For the neural controller, I used an open-source toolbox by the MIT Reliable Autonomous Systems Lab [4].

The state and input constraints were taken from a report defining the characteristics of an offshore wind turbine [5].

The controller was tested at air velocities ($V$) $3m/s, 5.5m/s, 8m/s$ and $10m/s$.

The optimal power-producing angular velocity is found using the equation below [6], where $R$ is the rotor radius.

$$
  \Omega_{ref} = \frac{V*7}{R}
$$

The tip to wind speed ratio is taken as 7, as this is approximately where the peak power is for a variety of blade angles (denoted as $\beta$ here), see figure 1.

<p align="center">
  <kbd>
    <img src="https://raw.githubusercontent.com/keatinl1/nl_mpc_wind_turbine/main/figs/pwr_vs_lambda.png">
  </kbd>
</p>
<p align="center">
Figure 1: Power coefficient vs. tip-speed-ratio [7].
</p>

## Results

The nominal NMPC with a terminal set required a short horizon and still yielded stable performance, making it a more efficient and feasible option. 

The tube-based robust NMPC was able to handle bounded disturbances, which is advantageous given the uncertain nature of wind. But this was at the cost of higher computation because of the need to solve two NMPC problems.

The neural controller was resource-intensive to train, performed poorly in practice, violating constraints in some instances. Access to high-performance compute resources would enable more and faster training; without this, neural control is not a reliable alternative.

<p align="center">
  <kbd>
    <img src="https://raw.githubusercontent.com/keatinl1/nl_mpc_wind_turbine/main/figs/results.png">
  </kbd>
</p>
<p align="center">
Figure 2: Angular velocity achieved by different controllers.
</p>

## Future work

Another potential direction within NMPC is a nominal controller with set constraints which are tightened, as is done in the nominal controller within tube-based NMPC. This controller could potentially offer a balance between robustness and computational load.


### References
[1] -Lars Henriksen. Nonlinear model predictive control of a simplified wind turbine. volume 44, pages 551–556, 08 2011.

[2] - Robin Verschueren, Gianluca Frison, Dimitris Kouzoupis, Jonathan Frey, Niels van Duijkeren, Andrea Zanelli, Branimir Novoselnik, Thivaharan Albin, Rien Quirynen, and Moritz Diehl. acados – a modular open-source framework for fast embedded optimal control. Mathematical Programming Computation, 2021.

[3] - M. Herceg, M. Kvasnica, C.N. Jones, and M. Morari. Multi-Parametric Toolbox 3.0. In Proc. of the European Control Conference, pages 502–510, Zürich, Switzerland, July 17–19 2013. http://control.ee.ethz.ch/~mpt.

[4] - Charles Dawson, Zengyi Qin, Sicun Gao, and Chuchu Fan. Safe nonlinear control using robust neural Lyapunov-barrier functions. In 5th Annual Conference on Robot Learning, 2021.

[5] - J Jonkman, S Butterfield, W Musial, and G Scott. Definition of a 5-mw reference wind turbine for offshore system development. Technical report, National Renewable Energy Lab. (NREL), Golden, CO (United States), 02 2009.

[6] - Magdi Ragheb. Optimal rotor tip speed ratio. https://users.wpi.edu/~cfurlong/me3320/DProject/Ragheb_OptTipSpeedRatio2014.pdf, 2014. Lecture notes of Course no. NPRE475, Accessed: 2025-06-20.

[7] - Aljodah, Ammar & Alwan, Marwah. (2021). Robust Speed Control Methodology for Variable Speed Wind Turbines. 10.48550/arXiv.2106.07022. 
