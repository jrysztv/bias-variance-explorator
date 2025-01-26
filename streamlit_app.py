import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bias_variance_explorator.simulation_utils.car_price_simulator import (
    CarPriceSimulator,
)
from bias_variance_explorator.simulator import MultipleRunsBiasVarianceExperiment
from bias_variance_explorator.viz_utils.viz_utils import BiasVarianceVisualization

# Define color scheme for consistency
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
if "experiment" not in st.session_state:
    st.session_state.experiment = None

if "visualization" not in st.session_state:
    st.session_state.visualization = None

# Set up the Streamlit app
st.title("Car Price Simulation: Fixed test set, repeated traning set")

# Sidebar
st.sidebar.header("Experiment Parameters")

with st.sidebar.expander("Experiment Setup", expanded=True):
    model_type = st.selectbox("Model Type", ["linear", "quadratic"], index=0)
    train_n_samples = st.number_input(
        "Training Samples per Run", value=100, min_value=50, max_value=5000, step=50
    )
    n_test = st.number_input(
        "Test Set Size", value=10000, min_value=100, max_value=200000, step=100
    )
    n_runs = st.number_input(
        "Number of Runs", value=20, min_value=1, max_value=500, step=1
    )
    train_seed_start = st.number_input("Training Seed", value=0, min_value=0, step=1)
    test_seed = st.number_input("Test Seed", value=999, min_value=0, step=1)
    noise_std = st.slider(
        "Noise Standard Deviation", min_value=10, max_value=500, value=100, step=10
    )

# Car price simulation parameters
with st.sidebar.expander("Car Price Simulation Parameters", expanded=False):
    base_price = st.number_input(
        "Base Price ($)", value=30000, min_value=5000, max_value=60000, step=1000
    )
    st.write("### Car age parameters:")
    age_mean = st.slider(
        "Mean Car Age (years)", min_value=0.0, max_value=15.0, value=5.0, step=0.5
    )
    age_std = st.slider(
        "Age Standard Deviation", min_value=0.0, max_value=5.0, value=2.0, step=0.5
    )
    age_coefficient = st.slider(
        "Age Coefficient", min_value=-3000, max_value=0, value=-500, step=100
    )
    age_quadratic_coefficient = st.slider(
        "Age Quadratic Coefficient",
        min_value=-50.0,
        max_value=50.0,
        value=-5.0,
        step=0.1,
    )
    age_exponent = st.slider(
        "Age Exponent", min_value=0.5, max_value=2.0, value=2.0, step=0.1
    )

    st.write("### Car mileage parameters:")
    mileage_mean = st.slider(
        "Mean Mileage (km)", min_value=0, max_value=250000, value=50000, step=5000
    )
    mileage_std = st.slider(
        "Mileage Std. Deviation",
        min_value=1000,
        max_value=50000,
        value=15000,
        step=5000,
    )
    mileage_coefficient = st.slider(
        "Mileage Coefficient (per scaled down unit)",
        min_value=-5.0,
        max_value=0.0,
        value=-1.0,
        step=0.1,
    )
    mileage_quadratic_coefficient = st.slider(
        "Mileage Quadratic Coefficient (per scaled down unit)²",
        min_value=-50.0,
        max_value=0.0,
        value=-5.0,
        step=0.5,
    )
    mileage_exponent = st.slider(
        "Mileage Exponent", min_value=0.5, max_value=2.0, value=2.0, step=0.1
    )
    mileage_scaledown = st.slider(
        "Mileage Scale Down Factor (how much to divide mileage (in km) by)",
        min_value=500,
        max_value=20000,
        value=11000,
        step=500,
    )

# Button to run experiment
if st.sidebar.button("Run Experiment"):
    with st.spinner("Running the multiple-runs bias-variance experiment..."):
        experiment = MultipleRunsBiasVarianceExperiment(
            n_test=n_test,
            test_seed=test_seed,
            n_runs=n_runs,
            train_n_samples=train_n_samples,
            train_seed_start=train_seed_start,
            noise_std=noise_std,
            model_type=model_type,
            # Pass all the simulator params
            base_price=base_price,
            age_mean=age_mean,
            age_std=age_std,
            mileage_mean=mileage_mean,
            mileage_std=mileage_std,
            age_coefficient=age_coefficient,
            mileage_coefficient=mileage_coefficient,
            age_quadratic_coefficient=age_quadratic_coefficient,
            mileage_quadratic_coefficient=mileage_quadratic_coefficient,
            age_exponent=age_exponent,
            mileage_exponent=mileage_exponent,
            mileage_scaledown=mileage_scaledown,
        )

        experiment.run_experiment()
        experiment.compute_metrics()

        visualization = BiasVarianceVisualization(experiment)

        # Save in session state
        st.session_state.experiment = experiment
        st.session_state.visualization = visualization

    st.success("Experiment completed!")

# Main area: show results if we have an experiment
if st.session_state.experiment is not None:
    experiment = st.session_state.experiment
    visualization = st.session_state.visualization

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.header("Cumulative average of metrics over number of runs")
        fig_metrics = visualization.plot_bias_variance_decomposition()
        st.pyplot(fig_metrics)

    with right_col:
        st.header("A sneak peek at the test set")
        # Slider to select number of observations to display in scatter plot
        n_observations = st.slider(
            "Number of Observations to Display",
            min_value=100,
            max_value=len(experiment.X_test),
            value=1000,
            step=100,
        )

        test_data = pd.DataFrame(
            {
                "mileage": experiment.X_test[:n_observations, 0],
                "age": experiment.X_test[:n_observations, 1],
                "price": experiment.y_test[:n_observations],
            }
        )
        fig_scatter = visualization.plot_scatter(test_data)
        st.pyplot(fig_scatter)

else:
    st.write("Use the sidebar to configure parameters, then click **Run Experiment**.")
