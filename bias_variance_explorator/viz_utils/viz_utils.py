import matplotlib.pyplot as plt


class BiasVarianceVisualization:
    """
    Handles visualization for Bias^2, Variance, and MSE terms, and scatter plots for simulation data.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    def plot_bias_variance_decomposition(self, simulation_id):
        """
        Plot a bar graph separating MSE, Bias^2, and Variance for the given simulation ID.
        Show another bar with the sum of Bias^2, Variance, and Noise.
        """
        if any(
            x is None
            for x in [self.experiment.bias2_list, self.experiment.variance_list]
        ):
            raise ValueError(
                "Metrics not computed! Call compute_cumulative_metrics() first."
            )

        if simulation_id < 1 or simulation_id > self.experiment.n_runs:
            raise ValueError("Invalid simulation ID. Must be between 1 and n_runs.")

        idx = simulation_id - 1
        bias2 = self.experiment.bias2_list[idx]
        variance = self.experiment.variance_list[idx]
        mse = self.experiment.mse_list[idx]
        noise = self.experiment.sigma2

        # Bar plot
        labels = ["MSE", "Bias^2 + Var + Noise"]
        mse_parts = [bias2, variance, noise]
        total = [sum(mse_parts)]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels[0], mse, label="MSE")
        ax.bar(labels[1], total, label="Bias^2 + Var + Noise")

        # Stacked bar for MSE breakdown
        ax.bar(labels[0], bias2, label="Bias^2")
        ax.bar(labels[0], variance, bottom=bias2, label="Variance")

        ax.set_ylabel("Error Terms")
        ax.set_title(f"Simulation ID: {simulation_id}")
        ax.legend()
        plt.show()

    def scatter_plot(self, simulation_id, data):
        """
        Plot scatter plots for price vs mileage and price vs age for the selected simulation ID.
        """
        if simulation_id < 1 or simulation_id > len(data):
            raise ValueError("Invalid simulation ID. Must match available dataset IDs.")

        # Select the correct DataFrame based on simulation_id
        selected_data = data[simulation_id - 1]  # simulation_id is 1-based

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Scatter plot: Price vs Mileage
        axes[0].scatter(selected_data["mileage"], selected_data["price"], alpha=0.7)
        axes[0].set_title("Price vs Mileage")
        axes[0].set_xlabel("Mileage")
        axes[0].set_ylabel("Price")

        # Scatter plot: Price vs Age
        axes[1].scatter(
            selected_data["age"], selected_data["price"], alpha=0.7, color="orange"
        )
        axes[1].set_title("Price vs Age")
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Price")

        plt.tight_layout()
        plt.show()
