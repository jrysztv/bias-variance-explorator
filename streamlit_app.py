import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from sklearn.linear_model import LinearRegression
from bias_variance_explorator.simulation_utils.car_price_simulator import (
    CarPriceSimulator,
)
from bias_variance_explorator.simulator import BiasVarianceExperiment

COLOR_SCHEME = {
    "MSE": "#1f77b4",  # blue
    "Bias^2": "#ff7f0e",  # orange
    "Variance": "#2ca02c",  # green
    "Noise": "#d62728",  # red
    "Bias^2 + Var + Noise": "#9467bd",  # purple
    "Price vs Mileage": "#17becf",  # cyan
    "Price vs Age": "#e377c2",  # pink
}

# Set the layout to wide
st.set_page_config(layout="wide")

# Initialize session state variables
# Instead of manually setting st.session_state.simulation_id here,
# we rely on the slider's 'key' argument to manage the state for us.
if "experiment" not in st.session_state:
    st.session_state.experiment = None

if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = None

# Set up the Streamlit app
st.title("Car Price Simulation: Bias-Variance Exploration")

# Sidebar input sections
st.sidebar.header("Simulation Setup")

with st.sidebar.expander("Experiment Parameters", expanded=True):
    model_type = st.selectbox("Model Type", options=["linear", "quadratic"], index=0)

    n_test = st.number_input(
        "Number of Test Samples", value=100, min_value=50, max_value=2000, step=50
    )
    n_runs = st.slider("Number of Runs", min_value=1, max_value=200, value=50, step=1)
    noise_std = st.slider(
        "Noise Standard Deviation", min_value=50, max_value=1000, value=100, step=50
    )
    train_n_samples = st.number_input(
        "Training Samples per Run", value=100, min_value=50, max_value=5000, step=100
    )
    train_seed_start = st.number_input(
        "Training Seed", value=0, min_value=0, max_value=1000, step=1
    )
    test_seed = st.number_input(
        "Test Set Seed", value=999, min_value=0, max_value=1000, step=1
    )

with st.sidebar.expander("Car Price Simulation Parameters", expanded=False):
    base_price = st.number_input(
        "Base Price ($)", value=30000, min_value=5000, max_value=60000, step=1000
    )
    st.write("# Car age parameters:")
    age_mean = st.slider(
        "Mean Car Age (years)", min_value=0.0, max_value=15.0, value=5.0, step=0.5
    )
    age_std = st.slider(
        "Age Standard Deviation", min_value=0.0, max_value=5.0, value=2.0, step=0.5
    )
    age_coefficient = st.slider(
        "Age Coefficient", min_value=-100, max_value=0, value=-500, step=10
    )
    age_quadratic_coefficient = st.slider(
        "Age Quadratic Coefficient", min_value=-50, max_value=50, value=-50, step=5
    )
    age_exponent = st.slider(
        "Age Exponent", min_value=0.5, max_value=2.0, value=2.0, step=0.1
    )
    # Mileage coefficients (per 1,000 km)
    st.write("# Car mileage parameters:")
    mileage_mean = st.slider(
        "Mean Mileage (km)", min_value=0, max_value=250000, value=50000, step=5000
    )
    mileage_std = st.slider(
        "Mileage Standard Deviation",
        min_value=1000,
        max_value=50000,
        value=15000,
        step=5000,
    )
    mileage_coefficient = st.slider(
        "Mileage Coefficient (per 1,000 km)",
        min_value=-1.0,
        max_value=0.0,
        value=-1.0,
        step=0.01,
    )
    mileage_quadratic_coefficient = st.slider(
        "Mileage Quadratic Coefficient (per 1,000 km)²",
        min_value=-5.0,
        max_value=5.0,
        value=-2.5,
        step=0.5,
    )
    mileage_exponent = st.slider(
        "Mileage Exponent", min_value=0.5, max_value=2.0, value=2.0, step=0.1
    )

# Create or run the experiment when the button is pressed
if st.sidebar.button("Run Experiment"):
    with st.spinner("Running the experiment..."):
        # Create experiment
        experiment = BiasVarianceExperiment(
            n_test=n_test,
            test_seed=test_seed,
            n_runs=n_runs,
            noise_std=noise_std,
            train_n_samples=train_n_samples,
            train_seed_start=train_seed_start,
            age_mean=age_mean,
            age_std=age_std,
            mileage_mean=mileage_mean,
            mileage_std=mileage_std,
            base_price=base_price,
            age_coefficient=age_coefficient,
            mileage_coefficient=mileage_coefficient,
            age_quadratic_coefficient=age_quadratic_coefficient,
            mileage_quadratic_coefficient=mileage_quadratic_coefficient,
            age_exponent=age_exponent,
            mileage_exponent=mileage_exponent,
        )
        experiment.run_experiment(model_type=model_type)
        experiment.compute_cumulative_metrics()
        st.session_state.experiment = experiment

        # Generate simulation data using the simulator parameters
        st.session_state.simulation_data = [
            experiment.simulator.generate_data() for _ in range(experiment.n_runs)
        ]
    st.success("Experiment completed!")

# Check if the experiment has been run
if st.session_state.experiment is not None:
    experiment = st.session_state.experiment

    # Layout with two columns
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Main bias-variance plot
        st.header("Bias-Variance Decomposition Results")
        fig, ax = plt.subplots()
        xs = np.arange(1, experiment.n_runs + 1)
        ax.plot(
            xs,
            experiment.mse_list,
            label="MSE (vs. noisy y_test)",
            color=COLOR_SCHEME["MSE"],
        )
        ax.plot(
            xs,
            experiment.bias2_list,
            label="Bias^2 (vs. noise-free f)",
            color=COLOR_SCHEME["Bias^2"],
        )
        ax.plot(
            xs,
            experiment.variance_list,
            label="Variance",
            color=COLOR_SCHEME["Variance"],
        )
        ax.plot(
            xs,
            experiment.bias2_list + experiment.variance_list + experiment.sigma2,
            label="Bias^2 + Var + Noise",
            linestyle="--",
            color=COLOR_SCHEME["Bias^2 + Var + Noise"],
        )
        ax.set_xlabel("Number of Simulations (k)")
        ax.set_ylabel("Cumulative-Average Metric")
        ax.set_title("Bias-Variance Decomposition with Known True Function")
        ax.legend()
        st.pyplot(fig)

        # Results Table
        st.header(f"Averages of {n_runs} Simulations")
        results_df = pd.DataFrame(
            {
                "Metric": ["MSE", "Bias^2", "Variance", "Noise Variance (σ²)"],
                "Average Value": [
                    experiment.mse_list[-1],
                    experiment.bias2_list[-1],
                    experiment.variance_list[-1],
                    experiment.sigma2,
                ],
            }
        )
        st.table(results_df)

    with right_col:
        # Simulation slider and per-simulation visualizations
        st.header("Per-Simulation Visualizations")

        # Use a key to let Streamlit manage simulation_id automatically
        st.slider(
            "Select Simulation ID",
            min_value=1,
            max_value=experiment.n_runs,
            value=1,  # default value
            key="simulation_id",  # stored in st.session_state.simulation_id
        )

        # Bar plot
        st.subheader("Bias-Variance Decomposition Bar Plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        idx = st.session_state.simulation_id - 1
        bias2 = experiment.bias2_list[idx]
        variance = experiment.variance_list[idx]
        mse = experiment.mse_list[idx]
        noise = experiment.sigma2

        # We draw multiple bars:
        ax.bar(["MSE"], [mse], label="MSE", color=COLOR_SCHEME["MSE"])
        ax.bar(
            ["Bias^2 + Var + Noise"],
            [bias2 + variance + noise],
            label="Bias^2 + Var + Noise",
            color=COLOR_SCHEME["Bias^2 + Var + Noise"],
        )
        ax.bar(["MSE"], [bias2], label="Bias^2", color=COLOR_SCHEME["Bias^2"])
        ax.bar(
            ["MSE"],
            [variance],
            bottom=bias2,
            label="Variance",
            color=COLOR_SCHEME["Variance"],
        )

        ax.set_ylabel("Error Terms")
        ax.set_title(f"Simulation ID: {st.session_state.simulation_id}")
        ax.legend()
        st.pyplot(fig)

        # Scatter plot
        st.subheader("Scatter Plots for Simulation Data")
        selected_data = st.session_state.simulation_data[idx]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].scatter(
            selected_data["mileage"],
            selected_data["price"],
            alpha=0.7,
            color=COLOR_SCHEME["Price vs Mileage"],
        )
        axes[0].set_title("Price vs Mileage")
        axes[0].set_xlabel("Mileage")
        axes[0].set_ylabel("Price")
        axes[1].scatter(
            selected_data["age"],
            selected_data["price"],
            alpha=0.7,
            color=COLOR_SCHEME["Price vs Age"],
        )
        axes[1].set_title("Price vs Age")
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Price")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.write("Adjust parameters in the sidebar and click 'Run Experiment' to begin.")
