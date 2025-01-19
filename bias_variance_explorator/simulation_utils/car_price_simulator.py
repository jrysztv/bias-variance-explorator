# %%
import numpy as np
import pandas as pd


class CarPriceSimulator:
    def __init__(self, **kwargs):
        # Default parameters
        self.params = {
            "n_samples": 1000,
            "noise_std": 100,
            "seed": None,
            "age_mean": 5.0,
            "age_std": 2.0,
            "mileage_mean": 50000,
            "mileage_std": 15000,
            "base_price": 30000,
            "age_coefficient": -2000,
            "mileage_coefficient": -0.1,
            "age_exponent": 0.5,
            "mileage_exponent": 0.5,
        }
        self.params.update(kwargs)

    @property
    def noise_std(self):
        return self.params["noise_std"]

    def true_function(self, age, mileage):
        p = self.params
        return (
            p["base_price"]
            + p["age_coefficient"] * np.power(age, p["age_exponent"])
            + p["mileage_coefficient"] * np.power(mileage, p["mileage_exponent"])
        )

    def generate_data(self):
        p = self.params
        if p["seed"] is not None:
            np.random.seed(p["seed"])

        age = np.maximum(
            0, np.random.normal(p["age_mean"], p["age_std"], p["n_samples"])
        )
        mileage = np.maximum(
            0, np.random.normal(p["mileage_mean"], p["mileage_std"], p["n_samples"])
        )
        fvals = self.true_function(age, mileage)
        noise = np.random.normal(0, p["noise_std"], p["n_samples"])
        y = np.clip(fvals + noise, a_min=0, a_max=None)

        return pd.DataFrame(
            {"age": age, "mileage": mileage, "price": y, "fvals": fvals}
        )
