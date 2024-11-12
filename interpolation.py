"""Interpolation Problem Sheet."""

# %%
import numpy as np


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


# Define time range
t = np.linspace(0, 1, 100)  # time from 0 to 10 with 100 points

# Calculate f(t) for these parameters
f_values = f(t, a, b, c)
