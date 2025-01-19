# %%
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

from simulation_utils.car_price_simulator import CarPriceSimulator
from viz_utils.viz_utils import BiasVarianceVisualization


class BiasVarianceExperiment:
    """
    Manages a bias-variance decomposition experiment:
      - fixed test set
      - repeated new training sets
      - accumulation of predictions
      - final plotting and numeric results
    """

    def __init__(
        self,
        n_test=2000,
        test_seed=999,
        n_runs=50,
        noise_std=100,
        train_n_samples=3000,
        train_seed_start=0,
        simulator=None,
    ):
        """
        :param n_test: number of samples in the *fixed* test set
        :param test_seed: seed for generating the fixed test set
        :param n_runs: how many times we re-generate a training set and fit
        :param noise_std: standard deviation of the noise
        :param train_n_samples: how many samples in each new training set
        :param train_seed_start: used as base for seeding each run
        """
        self.n_test = n_test
        self.test_seed = test_seed
        self.n_runs = n_runs
        self.noise_std = noise_std
        self.train_n_samples = train_n_samples
        self.train_seed_start = train_seed_start

        # We'll fill these after we run the experiment
        self.X_test = None
        self.y_test = None
        self.f_test = None
        self.sigma2 = noise_std**2

        self.all_predictions = None  # shape (n_runs, n_test)
        self.mse_list = None
        self.bias2_list = None
        self.variance_list = None

    def run_experiment(self):
        """
        1) Generate one fixed test set, storing both noise-free f_test and noisy y_test.
        2) For each run, generate fresh training data, fit, predict on fixed X_test.
        3) Store predictions in all_predictions for later analysis.
        """
        # 1) Build one fixed test set
        test_sim = CarPriceSimulator(
            n_samples=self.n_test, noise_std=self.noise_std, seed=self.test_seed
        )
        test_data = test_sim.generate_data()
        self.X_test = test_data[["age", "mileage"]].values
        self.y_test = test_data["price"].values  # single noisy realization
        self.f_test = test_data["fvals"].values  # noise-free
        self.sigma2 = test_sim.noise_std**2  # known noise variance

        # 2) Repeatedly generate new training sets and fit
        all_preds = []
        for run_id in range(self.n_runs):
            # Unique seed for each run
            seed_for_run = self.train_seed_start + run_id

            train_sim = CarPriceSimulator(
                n_samples=self.train_n_samples,
                noise_std=self.noise_std,
                seed=seed_for_run,
            )
            train_data = train_sim.generate_data()
            X_train = train_data[["age", "mileage"]].values
            y_train = train_data["price"].values

            # Fit a linear model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on the fixed test set
            preds = model.predict(self.X_test)
            all_preds.append(preds)

        self.all_predictions = np.array(all_preds)  # shape (n_runs, n_test)

    def compute_cumulative_metrics(self):
        """
        Compute the *cumulative average* bias^2, variance, MSE for k=1..n_runs.
        We'll store them in self.mse_list, self.bias2_list, self.variance_list.
        """
        mse_list = []
        bias2_list = []
        variance_list = []

        # Check that we have predictions
        if self.all_predictions is None:
            raise ValueError("No predictions yet! Call run_experiment() first.")

        for k in range(1, self.n_runs + 1):
            partial_preds = self.all_predictions[:k]  # shape (k, n_test)

            # -- BIAS^2 vs noise-free f(x) --
            avg_pred = partial_preds.mean(axis=0)  # mean across runs
            bias_sq = np.mean((avg_pred - self.f_test) ** 2)

            # -- VARIANCE --
            var_ = partial_preds.var(axis=0).mean()

            # -- MSE vs single noisy y_test --
            err = partial_preds - self.y_test[None, :]
            sq_err = err**2
            # first average over test samples => shape(k,)
            per_run_mse = sq_err.mean(axis=1)
            # then average across runs
            mse_val = per_run_mse.mean()

            mse_list.append(mse_val)
            bias2_list.append(bias_sq)
            variance_list.append(var_)

        self.mse_list = np.array(mse_list)
        self.bias2_list = np.array(bias2_list)
        self.variance_list = np.array(variance_list)

    def plot_results(self):
        """
        Plots the lines: MSE, Bias^2, Variance, and (Bias^2 + Var + noise).
        """
        if any(x is None for x in [self.mse_list, self.bias2_list, self.variance_list]):
            raise ValueError(
                "Metrics not computed! Call compute_cumulative_metrics() first."
            )

        xs = np.arange(1, self.n_runs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(xs, self.mse_list, label="MSE (vs. noisy y_test)")
        plt.plot(xs, self.bias2_list, label="Bias^2 (vs. noise-free f)")
        plt.plot(xs, self.variance_list, label="Variance")
        plt.plot(
            xs,
            self.bias2_list + self.variance_list + self.sigma2,
            label="Bias^2 + Var + Noise",
            linestyle="--",
        )

        plt.xlabel("Number of Simulations (k)")
        plt.ylabel("Cumulative-Average Metric")
        plt.title("Bias-Variance Decomposition with Known True Function")
        plt.legend()
        plt.show()

    def print_final_values(self):
        """
        Prints final numeric results for MSE, Bias^2, Var, sigma^2, sum.
        """
        if any(x is None for x in [self.mse_list, self.bias2_list, self.variance_list]):
            raise ValueError(
                "Metrics not computed! Call compute_cumulative_metrics() first."
            )

        final_idx = -1  # last element
        print(f"Final MSE:      {self.mse_list[final_idx]:.2f}")
        print(f"Final Bias^2:   {self.bias2_list[final_idx]:.2f}")
        print(f"Final Variance: {self.variance_list[final_idx]:.2f}")
        print(f"Noise variance (sigma^2): {self.sigma2:.2f}")
        sum_bv = (
            self.bias2_list[final_idx] + self.variance_list[final_idx] + self.sigma2
        )
        print(f"Final (Bias^2 + Var + Noise): {sum_bv:.2f}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # You can vary these parameters as you wish
    experiment = BiasVarianceExperiment(
        n_test=1000,
        test_seed=999,
        n_runs=100,
        noise_std=100,  # noise level
        train_n_samples=3000,
        train_seed_start=0,
    )
    experiment.run_experiment()
    experiment.compute_cumulative_metrics()
    experiment.plot_results()
    experiment.print_final_values()

    # Visualize results
    visualization = BiasVarianceVisualization(experiment)
    simulation_id = 10  # Example simulation ID

    # Generate data
    simulator = CarPriceSimulator()
    simulation_data = [simulator.generate_data() for _ in range(experiment.n_runs)]

    # Plot Bias-Variance Decomposition
    visualization.plot_bias_variance_decomposition(simulation_id)

    # Plot Scatter
    visualization.scatter_plot(simulation_id, simulation_data)

# %%
