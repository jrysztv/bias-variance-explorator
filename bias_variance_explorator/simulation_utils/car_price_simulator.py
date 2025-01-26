# %%
import numpy as np
import pandas as pd


class CarPriceSimulator:
    def __init__(
        self,
        n_samples=1000,
        noise_std=100,
        seed=None,
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
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.seed = seed

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

    def true_function(self, age, mileage):
        mileage_scaled = mileage / self.mileage_scaledown
        return (
            self.base_price
            + self.age_coefficient * age
            + self.age_quadratic_coefficient * np.power(age, self.age_exponent)
            + self.mileage_coefficient * mileage_scaled
            + self.mileage_quadratic_coefficient
            * np.power(mileage_scaled, self.mileage_exponent)
        )

    def generate_data(self):
        """Generate age, mileage, add random noise => price, and store noise‐free fvals."""
        if self.seed is not None:
            np.random.seed(self.seed)

        age = np.maximum(
            0, np.random.normal(self.age_mean, self.age_std, self.n_samples)
        )
        mileage = np.maximum(
            0, np.random.normal(self.mileage_mean, self.mileage_std, self.n_samples)
        )

        # Noise‐free function values
        fvals = self.true_function(age, mileage)

        # Add random noise to get final price
        noise = np.random.normal(0, self.noise_std, self.n_samples)
        price = np.clip(fvals + noise, a_min=0, a_max=None)

        return {
            "age": age,
            "mileage": mileage,
            "price": price,  # noisy
            "fvals": fvals,  # noise‐free
        }


if __name__ == "__main__":
    simulator = CarPriceSimulator()
    data = simulator.generate_data()
    print(data.head())

# %%
