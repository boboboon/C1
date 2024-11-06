"""General Brownian Motion Simulator."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


class GBMSimulator:
    """General Brownian Motion Simulator class."""

    def __init__(self, y_0: float, mu: float, sigma: float) -> None:
        """Creates our simulator that can then simualate both.

        Args:
            y_0 (float): Our initial value for t=0
            mu (float): Mu whatever
            sigma (float): Sigma whatever
        """
        self.y_0 = y_0
        self.mu = mu
        self.sigma = sigma

    def simulate_path(self, t: float, n: int) -> tuple:
        """Simulates a Brownian motion path with drift and volatility.

        Args:
            t (float): Total time for simulation.
            n (int): Number of steps.

        Returns:
            tuple: time_steps, y_vals
        """
        dt = t / (n - 1)  # Calculate the time increment
        rng = np.random.default_rng()

        # Initialize y_vals with the starting value
        y_vals = [self.y_0]

        # Generate y_vals as a cumulative process
        for _ in range(1, n):
            # Gaussian increment with mean 0 and variance dt
            dw = rng.normal(0, np.sqrt(dt))
            # Update using the Brownian motion with drift
            y_next = y_vals[-1] + self.mu * dt + self.sigma * dw
            y_vals.append(y_next)

        return np.linspace(0, t, n), y_vals


def main() -> None:
    """Plots our stuff."""
    parser = argparse.ArgumentParser(description="General Brownian Motion Simulator CLI")
    parser.add_argument("--y0", type=float, default=1.0, help="Initial value y_0 for t=0")
    parser.add_argument("--mu", type=float, default=0.05, help="Drift coefficient")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility coefficient")
    parser.add_argument("--t", type=float, default=0.1, help="Time duration")
    parser.add_argument("--n", type=int, default=1000, help="Number of time steps")
    parser.add_argument(
        "--output",
        type=str,
        default="gbm_plot.png",
        help="Output filename for the plot",
    )

    args = parser.parse_args()

    simulator = GBMSimulator(y_0=args.y0, mu=args.mu, sigma=args.sigma)
    t_steps, path = simulator.simulate_path(t=args.t, n=args.n)

    plt.figure(figsize=(10, 6))
    plt.plot(t_steps, path)
    plt.xlabel("Time (t)")
    plt.ylabel("Value")
    plt.title("Geometric Brownian Motion Simulation")
    plt.grid(visible=True)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
