# viz_utils.py
import matplotlib.pyplot as plt
import numpy as np


class BiasVarianceVisualization:
    """
    Plots metrics from a MultipleRunsBiasVarianceExperiment.
    Also can plot scatter of the fixed test set.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    def plot_bias_variance_decomposition(self):
        """
        Plot MSE, Bias², Variance, and (Bias² + Var + Noise) as a function of
        the run index k=1..n_runs.
        """
        if (
            self.experiment.mse_list is None
            or self.experiment.bias2_list is None
            or self.experiment.variance_list is None
        ):
            raise RuntimeError(
                "Experiment metrics not computed. Call experiment.compute_metrics() first."
            )

        ks = np.arange(1, self.experiment.n_runs + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(ks, self.experiment.mse_list, label="MSE (vs. noisy y)", marker="o")
        ax.plot(ks, self.experiment.bias2_list, label="Bias²", marker="o")
        ax.plot(ks, self.experiment.variance_list, label="Variance", marker="o")
        ax.plot(
            ks,
            self.experiment.bias2_list
            + self.experiment.variance_list
            + self.experiment.sigma2,
            label="Bias² + Var + Noise",
            linestyle="--",
            color="purple",
        )

        ax.set_xlabel("Number of Runs (k)")
        ax.set_ylabel("Metric Value")
        ax.set_title("Bias–Variance Decomposition across Multiple Runs")
        ax.legend()
        ax.grid(True)

        return fig

    def plot_scatter(self, test_df, title_suffix=""):
        """
        Scatter plots for price vs. mileage and price vs. age for the entire test set.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Price vs. Mileage
        axes[0].scatter(test_df["mileage"], test_df["price"], alpha=0.7, color="blue")
        axes[0].set_title(f"Price vs. Mileage {title_suffix}")
        axes[0].set_xlabel("Mileage")
        axes[0].set_ylabel("Price")

        # Right: Price vs. Age
        axes[1].scatter(test_df["age"], test_df["price"], alpha=0.7, color="orange")
        axes[1].set_title(f"Price vs. Age {title_suffix}")
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Price")

        plt.tight_layout()
        return fig
