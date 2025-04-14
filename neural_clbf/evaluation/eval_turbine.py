import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController


matplotlib.use('TkAgg')


def plot_turbine():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/review/clbf_turb-v2.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Update parameters
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [
            [1e-6, 1e-6, 1e-6],
        ]
    )

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    # eval_plot_turbine()
    plot_turbine()
