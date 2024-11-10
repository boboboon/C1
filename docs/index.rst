GBM Simulator Documentation
==========================

Welcome to the GBM Simulator documentation! This package provides tools for simulating Geometric Brownian Motion (GBM), a fundamental stochastic process widely used in financial mathematics and other fields.

Mathematical Background
---------------------

Geometric Brownian Motion (GBM) is a continuous-time stochastic process where the logarithm of the randomly varying quantity follows a Brownian motion with drift. It is particularly useful for modeling quantities that:

* Cannot take negative values
* Have variations that tend to scale with the current value
* Show continuous changes over time

The GBM follows the stochastic differential equation:

.. math::

   dS = \mu S dt + \sigma S dW

where:

* :math:`S` is the quantity being modeled (e.g., stock price)
* :math:`\mu` (mu) is the drift coefficient, representing the expected return
* :math:`\sigma` (sigma) is the volatility coefficient
* :math:`dW` is a Wiener process increment

Common applications include:

* Stock price modeling in financial markets
* Asset valuation in option pricing
* Population growth models with random fluctuations
* Price modeling for commodities and other financial instruments

Installation
------------

To install GBM Simulator in development mode:

.. code-block:: bash

   git clone <https://github.com/boboboon/C1>
   cd C1
   pip install -e .

Usage
-----

The package provides both a command-line interface and a Python API for simulating GBM paths.

Command Line Interface
~~~~~~~~~~~~~~~~~~~~

Generate a GBM simulation directly from the command line:

.. code-block:: bash

   simulate --y0 1.0 --mu 0.05 --sigma 0.2 --t 0.1 --n 1000 --output gbm_plot.png

Parameters:

* ``--y0``: Initial value at t=0 (default: 1.0)
* ``--mu``: Drift coefficient, controlling the trend direction (default: 0.05)
* ``--sigma``: Volatility coefficient, controlling the magnitude of random fluctuations (default: 0.2)
* ``--t``: Total simulation time (default: 0.1)
* ``--n``: Number of time steps (default: 1000)
* ``--output``: Output plot filename (default: "gbm_plot.png")

Python Interface
~~~~~~~~~~~~~~

Use the simulator programmatically in Python:

.. code-block:: python

   from C1.gbm_simulator import GBMSimulator

   # Create a simulator instance
   simulator = GBMSimulator(
       y_0=1.0,    # Initial value
       mu=0.05,    # Drift coefficient
       sigma=0.2   # Volatility coefficient
   )
   
   # Generate a path
   t_steps, path = simulator.simulate_path(t=0.1, n=1000)

The simulation uses the Euler-Maruyama method for numerical approximation of the stochastic differential equation.

Understanding the Parameters
--------------------------

y_0 (Initial Value)
~~~~~~~~~~~~~~~~~~
The starting point of your simulation at t=0. This could represent:

* Initial stock price
* Starting asset value
* Initial population size

mu (Drift)
~~~~~~~~~
The drift parameter :math:`\mu` determines the overall trend:

* Positive values create an upward trend
* Negative values create a downward trend
* Larger absolute values create stronger trends

sigma (Volatility)
~~~~~~~~~~~~~~~~
The volatility parameter :math:`\sigma` controls the randomness:

* Higher values increase the randomness of the path
* Lower values make the path more deterministic
* Affects how much the path can deviate from the trend

Dependencies
-----------

Core dependencies:

* numpy: For numerical operations
* matplotlib: For visualization
* loguru: For logging

See ``pyproject.toml`` for a complete list of dependencies.

License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.