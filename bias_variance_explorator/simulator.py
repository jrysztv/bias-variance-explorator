# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from bias_variance_explorator.simulation_utils.car_price_simulator import (
    CarPriceSimulator,
)
from bias_variance_explorator.viz_utils.viz_utils import BiasVarianceVisualization


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
        age_mean=5.0,
        age_std=2.0,
        mileage_mean=50000,
        mileage_std=15000,
        base_price=30000,
        age_coefficient=-2000,
        mileage_coefficient=-0.1,
        age_quadratic_coefficient=0,
        mileage_quadratic_coefficient=0,
        age_exponent=0.5,
        mileage_exponent=0.5,
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

        self.age_mean = age_mean
        self.age_std = age_std
        self.mileage_mean = mileage_mean
        self.mileage_std = mileage_std
        self.base_price = base_price
        self.age_coefficient = age_coefficient
        self.mileage_coefficient = mileage_coefficient
        self.age_quadratic_coefficient = age_quadratic_coefficient
        self.mileage_quadratic_coefficient = mileage_quadratic_coefficient
        self.age_exponent = age_exponent
        self.mileage_exponent = mileage_exponent

        # We'll fill these after we run the experiment
        self.X_test = None
        self.y_test = None
        self.f_test = None
        self.sigma2 = noise_std**2

        self.all_predictions = None  # shape (n_runs, n_test)
        self.mse_list = None
        self.bias2_list = None
        self.variance_list = None
        self.simulator = None

        self.create_simulator()

    def create_simulator(self):
        """
        Create a simulator with consistent parameters for generating the test set.
        """
        self.simulator = CarPriceSimulator(
            n_samples=self.n_test,
            noise_std=self.noise_std,
            seed=self.test_seed,
            age_mean=self.age_mean,
            age_std=self.age_std,
            mileage_mean=self.mileage_mean,
            mileage_std=self.mileage_std,
            base_price=self.base_price,
            age_coefficient=self.age_coefficient,
            mileage_coefficient=self.mileage_coefficient,
            age_quadratic_coefficient=self.age_quadratic_coefficient,
            mileage_quadratic_coefficient=self.mileage_quadratic_coefficient,
            age_exponent=self.age_exponent,
            mileage_exponent=self.mileage_exponent,
        )

    def run_experiment(self, model_type="linear"):
        """
        Run the bias-variance experiment for either linear or quadratic regression.
        Ensures that both training and test datasets are generated with consistent parameters.
        """
        # 1) Build one fixed test set
        test_sim = (
            self.simulator
        )  # This simulator includes all user-specified parameters
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

            # Create a new training simulator with consistent parameters
            train_sim = CarPriceSimulator(
                n_samples=self.train_n_samples,
                noise_std=self.noise_std,
                seed=seed_for_run,
                age_mean=self.age_mean,
                age_std=self.age_std,
                mileage_mean=self.mileage_mean,
                mileage_std=self.mileage_std,
                base_price=self.base_price,
                age_coefficient=self.age_coefficient,
                mileage_coefficient=self.mileage_coefficient,
                age_quadratic_coefficient=self.age_quadratic_coefficient,
                mileage_quadratic_coefficient=self.mileage_quadratic_coefficient,
                age_exponent=self.age_exponent,
                mileage_exponent=self.mileage_exponent,
            )

            # Generate training data
            train_data = train_sim.generate_data()
            X_train = train_data[["age", "mileage"]].values
            y_train = train_data["price"].values

            # Fit a model
            if model_type == "linear":
                model = LinearRegression()
            elif model_type == "quadratic":
                model = Pipeline(
                    [
                        ("poly_features", PolynomialFeatures(degree=2)),
                        ("linear_regression", LinearRegression()),
                    ]
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            model.fit(X_train, y_train)
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


class IncrementalTestExperiment:
    def __init__(
        self,
        train_n_samples=500,
        train_seed=42,
        test_size_start=100,
        test_size_end=1000,
        test_step=100,
        test_seed=999,
        noise_std=100,
        model_type="linear",
        **simulator_params,
    ):
        """
        Initialize an experiment to evaluate performance metrics incrementally
        with increasing test set sizes.
        """
        self.train_n_samples = train_n_samples
        self.train_seed = train_seed
        self.test_size_start = test_size_start
        self.test_size_end = test_size_end
        self.test_step = test_step
        self.test_seed = test_seed
        self.noise_std = noise_std
        self.model_type = model_type

        self.simulator_params = simulator_params
        self.train_simulator = None
        self.test_simulator = None
        self.model = None
        self.results = {"mse": [], "bias2": [], "variance": [], "test_sizes": []}

    def setup_data(self):
        """
        Set up simulators for training and the largest test set.
        """
        self.train_simulator = CarPriceSimulator(
            n_samples=self.train_n_samples,
            noise_std=self.noise_std,
            seed=self.train_seed,
            **self.simulator_params,
        )
        self.test_simulator = CarPriceSimulator(
            n_samples=self.test_size_end,
            noise_std=self.noise_std,
            seed=self.test_seed,
            **self.simulator_params,
        )

    def fit_model(self):
        """
        Train a linear or quadratic regression model using simulated data.
        """
        train_data = self.train_simulator.generate_data()
        X_train = train_data[["age", "mileage"]].values
        y_train = train_data["price"].values

        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "quadratic":
            self.model = Pipeline(
                [
                    ("poly_features", PolynomialFeatures(degree=2)),
                    ("regressor", LinearRegression()),
                ]
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        self.model.fit(X_train, y_train)

    def run_incremental_tests(self):
        """
        Evaluate the model incrementally with increasing test set sizes.
        """
        test_data = self.test_simulator.generate_data()
        X_test_full = test_data[["age", "mileage"]].values
        y_test_full = test_data["price"].values
        y_true_full = test_data["fvals"].values

        for size in range(self.test_size_start, self.test_size_end + 1, self.test_step):
            X_test = X_test_full[:size]
            y_test = y_test_full[:size]
            y_true = y_true_full[:size]

            y_pred = self.model.predict(X_test)
            mse = np.mean((y_pred - y_test) ** 2)
            bias2 = np.mean((np.mean(y_pred) - y_true) ** 2)
            variance = np.var(y_pred)

            self.results["mse"].append(mse)
            self.results["bias2"].append(bias2)
            self.results["variance"].append(variance)
            self.results["test_sizes"].append(size)

    def run(self):
        """
        Orchestrates the experiment by setting up data, training the model, and testing incrementally.
        """
        self.setup_data()
        self.fit_model()
        self.run_incremental_tests()


class MultipleRunsBiasVarianceExperiment:
    """
    Runs a classical bias–variance decomposition:
      1) Generate ONE *fixed* test set of size n_test.
      2) For each of n_runs:
           - Generate a new, independent training set
           - Fit model
           - Predict on the same test set
      3) Compute average predictions, compute bias^2 & variance, compare vs. MSE.
    """

    def __init__(
        self,
        n_test=1000,
        test_seed=999,
        n_runs=50,
        train_n_samples=3000,
        train_seed_start=0,
        noise_std=100,
        model_type="linear",
        # Any additional simulator params (age_mean, mileage_mean, etc.) can go here:
        age_mean=5.0,
        age_std=2.0,
        mileage_mean=50000,
        mileage_std=15000,
        base_price=30000,
        age_coefficient=-2000,
        mileage_coefficient=-0.1,
        age_quadratic_coefficient=0,
        mileage_quadratic_coefficient=0,
        age_exponent=0.5,
        mileage_exponent=0.5,
        mileage_scaledown=10000,
    ):
        self.n_test = n_test
        self.test_seed = test_seed
        self.n_runs = n_runs
        self.train_n_samples = train_n_samples
        self.train_seed_start = train_seed_start
        self.noise_std = noise_std
        self.model_type = model_type

        # Store extra simulator parameters for consistent usage
        self.age_mean = age_mean
        self.age_std = age_std
        self.mileage_mean = mileage_mean
        self.mileage_std = mileage_std
        self.base_price = base_price
        self.age_coefficient = age_coefficient
        self.mileage_coefficient = mileage_coefficient
        self.age_quadratic_coefficient = age_quadratic_coefficient
        self.mileage_quadratic_coefficient = mileage_quadratic_coefficient
        self.age_exponent = age_exponent
        self.mileage_exponent = mileage_exponent

        self.mileage_scaledown = mileage_scaledown

        # Outputs
        self.X_test = None
        self.y_test = None  # single noisy realization
        self.f_test = None  # underlying noise‐free f(x)
        self.sigma2 = noise_std**2

        self.all_predictions = None  # shape (n_runs, n_test)
        self.mse_list = None
        self.bias2_list = None
        self.variance_list = None

        self.models = []  # store models for each run

        # Build the one fixed test set immediately
        self._create_fixed_test_set()

    def _create_fixed_test_set(self):
        """
        Create one *fixed* test set with the specified size and seed.
        """
        test_sim = CarPriceSimulator(
            n_samples=self.n_test,
            noise_std=self.noise_std,
            seed=self.test_seed,
            age_mean=self.age_mean,
            age_std=self.age_std,
            mileage_mean=self.mileage_mean,
            mileage_std=self.mileage_std,
            base_price=self.base_price,
            age_coefficient=self.age_coefficient,
            mileage_coefficient=self.mileage_coefficient,
            age_quadratic_coefficient=self.age_quadratic_coefficient,
            mileage_quadratic_coefficient=self.mileage_quadratic_coefficient,
            age_exponent=self.age_exponent,
            mileage_exponent=self.mileage_exponent,
            mileage_scaledown=self.mileage_scaledown,
        )
        data = test_sim.generate_data()

        # Store test data as arrays
        self.X_test = np.column_stack((data["age"], data["mileage"]))
        self.y_test = data["price"]  # single noisy realization
        self.f_test = data["fvals"]  # noise‐free true function at the same points

    def run_experiment(self):
        """
        For each of the n_runs, generate a new training set, fit the chosen model,
        predict on the fixed X_test, and store predictions.
        """
        predictions = []
        for run_id in range(self.n_runs):
            seed_for_run = self.train_seed_start + run_id

            # Make a new training simulator with the same parameters, but new seed
            train_sim = CarPriceSimulator(
                n_samples=self.train_n_samples,
                noise_std=self.noise_std,
                seed=seed_for_run,
                age_mean=self.age_mean,
                age_std=self.age_std,
                mileage_mean=self.mileage_mean,
                mileage_std=self.mileage_std,
                base_price=self.base_price,
                age_coefficient=self.age_coefficient,
                mileage_coefficient=self.mileage_coefficient,
                age_quadratic_coefficient=self.age_quadratic_coefficient,
                mileage_quadratic_coefficient=self.mileage_quadratic_coefficient,
                age_exponent=self.age_exponent,
                mileage_exponent=self.mileage_exponent,
            )
            train_data = train_sim.generate_data()
            X_train = np.column_stack((train_data["age"], train_data["mileage"]))
            y_train = train_data["price"]

            # Choose model
            if self.model_type == "linear":
                model = LinearRegression()
            elif self.model_type == "quadratic":
                model = Pipeline(
                    [
                        # ("scaler", StandardScaler()),
                        ("poly_features", PolynomialFeatures(degree=2)),
                        ("regressor", LinearRegression()),
                    ]
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            # Fit and predict
            model.fit(X_train, y_train)
            self.models.append(model)
            preds = model.predict(self.X_test)
            predictions.append(preds)

        # Convert list to array of shape (n_runs, n_test)
        self.all_predictions = np.array(predictions)

    def compute_metrics(self):
        """
        Compute the final bias^2, variance, and MSE from all_predictions.
        """
        if self.all_predictions is None:
            raise RuntimeError("No predictions found. Did you call run_experiment()?")

        # Ensure y_test and f_test are NumPy arrays
        self.y_test = np.asarray(self.y_test)
        self.f_test = np.asarray(self.f_test)

        n_runs, n_test = self.all_predictions.shape
        mse_list = []
        bias2_list = []
        variance_list = []

        # Loop for cumulative computation up to run k
        for k in range(1, n_runs + 1):
            partial_preds = self.all_predictions[:k]  # shape (k, n_test)

            # Average prediction for each test point (mean across runs)
            avg_pred = np.mean(partial_preds, axis=0)  # shape (n_test,)

            # Bias^2: Average squared difference from the true f(x) across test points
            bias_sq = np.mean((avg_pred - self.f_test) ** 2)

            # Variance: Average variance of predictions across runs, per test point
            var_ = np.mean(np.var(partial_preds, axis=0))

            # MSE (vs noisy y_test): Compute squared errors, average over test points & runs
            sq_errors = (partial_preds - self.y_test[None, :]) ** 2  # shape (k, n_test)
            mse_val = np.mean(sq_errors)  # scalar

            mse_list.append(mse_val)
            bias2_list.append(bias_sq)
            variance_list.append(var_)

        self.mse_list = np.array(mse_list)
        self.bias2_list = np.array(bias2_list)
        self.variance_list = np.array(variance_list)

    def print_results(self):
        """
        Print final bias^2, variance, MSE, and check that MSE ≈ bias^2 + variance + sigma^2
        (the noise variance).
        """
        if any(x is None for x in [self.mse_list, self.bias2_list, self.variance_list]):
            raise RuntimeError("Metrics not computed. Call compute_metrics() first.")

        # Last index => final result using all n_runs
        final_idx = -1
        mse = self.mse_list[final_idx]
        b2 = self.bias2_list[final_idx]
        var_ = self.variance_list[final_idx]
        noise = self.sigma2
        sum_bv = b2 + var_ + noise

        print(f"Final results (after {self.n_runs} runs):")
        print(f"  MSE             = {mse:.3f}")
        print(f"  Bias^2          = {b2:.3f}")
        print(f"  Variance        = {var_:.3f}")
        print(f"  Noise variance  = {noise:.3f}")
        print(f"  B^2 + Var + σ^2 = {sum_bv:.3f}")

    def plot_results(self):  # legacy
        """
        Plot MSE, bias^2, variance, and (bias^2 + variance + sigma^2) as a function of
        the cumulative run index k = 1..n_runs.
        """
        if any(x is None for x in [self.mse_list, self.bias2_list, self.variance_list]):
            raise RuntimeError("Metrics not computed. Call compute_metrics() first.")

        ks = np.arange(1, self.n_runs + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(ks, self.mse_list, label="MSE (vs. noisy test y)")
        plt.plot(ks, self.bias2_list, label="Bias²")
        plt.plot(ks, self.variance_list, label="Variance")
        plt.plot(
            ks,
            self.bias2_list + self.variance_list + self.sigma2,
            label="Bias² + Variance + Noise",
            linestyle="--",
        )
        plt.xlabel("Number of Runs (k)")
        plt.ylabel("Metric Value")
        plt.title("Bias–Variance Decomposition")
        plt.legend()
        plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    experiment = MultipleRunsBiasVarianceExperiment(
        n_test=10000,
        test_seed=999,
        n_runs=50,
        train_n_samples=100,
        train_seed_start=0,
        noise_std=100,
        model_type="quadratic",  # or "linear"
        age_mean=5.0,
        age_std=2.0,
        mileage_mean=50000,
        mileage_std=15000,
        base_price=30000,
        age_coefficient=-2000,
        mileage_coefficient=-0.1,
        age_quadratic_coefficient=-5,
        mileage_quadratic_coefficient=-5,
        age_exponent=2,
        mileage_exponent=2,
    )

    # Run the entire experiment (train once, test incrementally)
    experiment.run_experiment()  # builds multiple training sets & fits
    experiment.compute_metrics()  # calculates bias^2, variance, MSE
    experiment.print_results()  # prints final numeric results
    experiment.plot_results()  # optional: plots MSE, bias^2, var, etc.

# %%
