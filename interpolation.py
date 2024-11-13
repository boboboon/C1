"""Interpolation Problem Sheet."""

# %%
# ? Question 1
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import FloatSlider, interact
from memory_profiler import memory_usage
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.stats import qmc


def f(t: float, a: float, b: float, c: float) -> float:
    """Outputs our func.

    Args:
        t (float): _description_
        a (float): _description_
        b (float): _description_
        c (float): _description_

    Returns:
        float: _description_
    """
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)


# Example usage
# Define parameters
a = 0.1
b = -0.13
c = 9


t = np.linspace(0, 1, 100)

f_values = f(t, a, b, c)

linear_interp = interp1d(t, f_values, kind="linear")
cubic_interp = interp1d(t, f_values, kind="cubic")

# %%
# ? Question 2
# Define a new, finer time range for interpolation
t_fine = np.linspace(0, 1, 10000)  # time from 0 to 1 with 500 points

f_linear_interp = linear_interp(t_fine)
f_cubic_interp = cubic_interp(t_fine)
# %%
# ? Question 3
plt.figure()


plt.plot(t_fine, f_linear_interp, label="Linear")
plt.plot(t_fine, f_cubic_interp, label="Cubic")

f_actual = [f(t, a, b, c) for t in t_fine]
plt.plot(t_fine, f_actual, label="Actual")


plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.title("Comparison of Linear, Cubic Interpolation and Actual f(t)")
plt.show()
# %%
# ? Question 4
ratio_linear = f_linear_interp / f_actual
ratio_cubic = f_cubic_interp / f_actual

# Spikes are due to close to 0

plt.figure()


plt.plot(t_fine, ratio_linear, label="Linear")
plt.plot(t_fine, ratio_cubic, label="Cubic")


plt.xlabel("t")
plt.ylabel("r")
plt.legend()
plt.title("Comparison of Linear and Cubic to Actual")
plt.show()

# %%
# ? Question 5

# Consider now all paramaters fixed except a (and t).
# We assume the parameter a can take values between 0 and 1.
# Generate 10 samples of f (i.e., 10 time series) corresponding to linearly spaced values of
# spanning the interval.
# Store them in a pandas DataFrame and plot them with the plot method of the DataFrame.

a_range = np.linspace(0, 1, 11)

f_a_range = [[f(t, a, b, c) for t in t] for a in a_range]

a_df = pd.concat(
    {f"{a}": pd.Series([f(ti, a, b, c) for ti in t], index=t) for a in a_range},
    axis=1,
)

a_df.columns.name = "a"

# %%
# ? Question 6

# Create an interpolator that interpolates over a (same range as previous question)
# and returns the full time series (i.e., values of  for all time points) over the
# original time grid, i.e., .


# Step 1: Prepare the 2D grid of values (a, t)
# a_range and t represent the two dimensions over which we interpolate
f_values_grid = a_df.to_numpy().T  # Transpose to match shape (a, t) instead of (t, a)

# Create the interpolator over both a and t
interpolator = RegularGridInterpolator((a_range, t), f_values_grid, method="linear")


# ? Question 7
def plot_interpolated_vs_actual(a_target: float) -> None:
    """Plot interpolated vs actual and ratio in separate plots for a given `a` target."""
    # Get interpolated values for the target `a` over the original time grid `t`
    points = np.array([[a_target, t_i] for t_i in t])
    f_interp_at_a_target = interpolator(points)

    # Calculate actual values for comparison
    f_actual = [f(ti, a_target, b, c) for ti in t]

    # Plot 1: Interpolated vs Actual
    plt.figure(figsize=(10, 6))
    plt.plot(t, f_interp_at_a_target, label=f"Interpolated at a = {a_target}")
    plt.plot(t, f_actual, label="Actual", linestyle="--")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.title(f"Interpolated vs Actual for a = {a_target}")
    plt.show()

    # Plot 2: Ratio of Interpolated to Actual
    plt.figure(figsize=(10, 6))
    ratio = np.array(f_interp_at_a_target) / np.array(f_actual)
    plt.plot(t, ratio, label="Ratio (Interpolated / Actual)", linestyle="-.")
    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.legend()
    plt.title(f"Ratio of Interpolated to Actual for a = {a_target}")
    plt.show()


# Use ipywidgets to create a slider for `a` values
interact(
    plot_interpolated_vs_actual,
    a_target=FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="a"),
)

# Seems to suffer at max gradients
# %%
# ? Question 9
# We will now consider both  and  as interpolation parameters.

# Our interpolator should therefore interpolate accross both  and  ranges.

# Generate  parameter value pairs  in the range  and  using latin hyper cube sampling.

# Show the  and  samples as a 2D scatter plot.


# Define the parameter ranges for a and b
a_min, a_max = 0, 1
b_min, b_max = -0.5, 0.5

num_samples = 100
sampler = qmc.LatinHypercube(d=2)
lhs_samples = sampler.random(n=num_samples)
a_samples = lhs_samples[:, 0] * (a_max - a_min) + a_min
b_samples = lhs_samples[:, 1] * (b_max - b_min) + b_min

# %%
# ? Comparing to Uniform Samples (Question 10)
rng = np.random.default_rng()

a_samples_uniform = rng.uniform(0, 1, size=num_samples)
b_samples_uniform = rng.uniform(-0.5, 0.5, size=num_samples)

plt.figure(figsize=(8, 6))
plt.scatter(a_samples, b_samples, color="blue", label="LHS Samples")
plt.scatter(a_samples_uniform, b_samples_uniform, color="red", label="Uniform Samples")
plt.xlabel("a")
plt.ylabel("b")
plt.title("2D Scatter Plot of a and b Samples")
plt.legend()
plt.show()

# %%
lhs_df = pd.DataFrame({"a_samples": a_samples, "b_samples": b_samples})
uniform_df = pd.DataFrame({"a_samples": a_samples_uniform, "b_samples": b_samples_uniform})

# The mean and std is far more similar for uniform between a and b
# I am guessing this is because we used the same distribution to extract a and b (then transformed it)
# but for latin hyper cube thats not the case? Latin hypercube is stratified (to cover most spaces)

# %%
# ? Question 11
# Create the interpolator over the parameter space ,
# interpolating over samples of the function
# evaluated at the original time grid .


a_mesh, b_mesh, t_mesh = np.meshgrid(a_samples, b_samples, t, indexing="ij")
points = np.column_stack([t_mesh.ravel(), a_mesh.ravel(), b_mesh.ravel()])

values = np.array([f(t, a, b, c) for t, a, b in points])

new_a, new_b = 0.5, 0.5
new_points = np.array([(t, new_a, new_b) for t in t])

# Perform interpolation
a_b_interpolated = griddata(points, values, new_points, method="nearest")


# %%
# ? Question 12
# Setup figure and axes for plotting
def plot_interactive(new_a: float = 0.5, new_b: float = 0.5) -> None:
    """Interpolated values for current a and b from sliders.

    Args:
        new_a (float, optional): _description_. Defaults to 0.5.
        new_b (float, optional): _description_. Defaults to 0.5.
    """
    new_points = np.array([(t_val, new_a, new_b) for t_val in t])
    a_b_interpolated = griddata(points, values, new_points, method="nearest")

    # True values at these (t, a, b)
    true_values = np.array([f(t_val, new_a, new_b, c) for t_val in t])
    ratios = a_b_interpolated / true_values

    # Plotting
    plt.figure(figsize=(10, 8))

    # Subplot 1: Function Curves
    plt.subplot(2, 1, 1)
    for a_value in a_samples:
        for b_value in b_samples:
            f_values = [f(t_val, a_value, b_value, c) for t_val in t]
            plt.plot(t, f_values, color="r", alpha=0.3)  # Original curves
    plt.plot(t, a_b_interpolated, color="b", label=f"Interpolated (a={new_a:.2f}, b={new_b:.2f})")
    plt.title("Function Curves with Interpolated Curve Highlighted")

    # Subplot 2: Ratio Plot
    plt.subplot(2, 1, 2)
    plt.plot(t, ratios, color="purple")
    plt.title("Ratio of Interpolated Values to True Values")
    plt.xlabel("t")
    plt.ylabel("Interpolated / True")

    plt.tight_layout()
    plt.show()


# Setting up interactive sliders
a_slider = FloatSlider(
    value=0.5, min=min(a_samples), max=max(a_samples), step=0.01, description="a"
)
b_slider = FloatSlider(
    value=0.5, min=min(b_samples), max=max(b_samples), step=0.01, description="b"
)

# Using interact to link sliders and plot function
interact(plot_interactive, new_a=a_slider, new_b=b_slider)

# Ratio around 0 spikes as small imperfections will spike.


# %%
# ? Question 13
# Compare memory and time of (i) the original function call and (ii) the interpolator call.
def original_function_call(t: float, a: float, b: float, c: float) -> np.array:
    """Calls our original function nicely.

    Args:
        t (float): _description_
        a (float): _description_
        b (float): _description_
        c (float): _description_

    Returns:
        np.array: _description_
    """
    return np.array([f(t_val, a, b, c) for t_val in t])


# Interpolator call
def interpolator_call(new_a: int, new_b: int) -> np.array:
    """Calls our interpolator instead.

    Args:
        new_a (int): _description_
        new_b (int): _description_

    Returns:
        np.array: _description_
    """
    new_points = np.array([(t_val, new_a, new_b) for t_val in t])
    return griddata(points, values, new_points, method="nearest")


time_original = timeit(lambda: original_function_call(t, new_a, new_b, c), number=10)
time_interpolator = timeit(lambda: interpolator_call(new_a, new_b), number=10)

memory_original = memory_usage((original_function_call, (t, new_a, new_b, c)), interval=0.01)
memory_interpolator = memory_usage((interpolator_call, (new_a, new_b)), interval=0.01)
# %%
