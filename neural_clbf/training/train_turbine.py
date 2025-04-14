from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint


from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import Turbine
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutTimeSeriesExperiment
)
from neural_clbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [1e-6, 1e-6, 1e-6],
        [1e-6, 1e-6, -100],
        [1e-6, 1e-6, 100],
        [1e-6, -5.0, 1e-6],
        [1.3, 1e-6, 1e-6]
    ]
)
simulation_dt = 0.05


def main(args):
    # Define the scenarios
    nominal_params = {"R": 61.5, "I": 11776047.0*3, "p": 1.225, "V":5.5}
    scenarios = [
        nominal_params
    ]

    # Define the dynamics model
    dynamics_model = Turbine(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (1e-6, 1.27),      # OMEGA: full range
        (1e-6, 90.0),      # THETA: full range
        (-47.0, 47.0),    # QG: full range
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(0.0, 90.0), (0.0, 90.0)],
        n_grid=30,
        x_axis_index=Turbine.OMEGA,
        y_axis_index=Turbine.QG,
        x_axis_label="$\\Omega$",
        y_axis_label="$Q_g$",
        plot_unsafe_region=False,
    )

    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        [Turbine.OMEGA, Turbine.THETA, Turbine.QG],
        ["$\\Omega$","$\\theta$", "$Q_g$"],
        [Turbine.U1, Turbine.U2],
        ["$\\theta_dot$", "$Q_gdot$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )

    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/turbine/",  # Folder to save checkpoints
        filename="clbf_turb",  # Customize if needed
        save_top_k=1,  # Save top 3 checkpoints
        monitor="Total loss / val",  # Make sure your model logs this
        mode="min",  # Lower val_loss = better
        save_last=True,  # Always save the last epoch too
    )

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=True,
        disable_gurobi=True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/turbine",
        name=f"commit_{current_git_hash()}",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=1,
        gradient_clip_val=100.0,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
