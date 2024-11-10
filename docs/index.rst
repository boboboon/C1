GBM Simulator Documentation
==========================

Welcome to the GBM Simulator documentation! This documentation provides an overview of the General Brownian Motion (GBM) Simulator project, which simulates paths of a geometric Brownian motion with user-defined parameters.

Installation
------------

To install GBM Simulator in development mode:

.. code-block:: bash

   git clone <repository-url>
   cd C1
   pip install -e .

Features
--------

- Simulate Geometric Brownian Motion paths with customizable parameters
- Command-line interface for quick simulations
- Matplotlib visualization of the simulated paths
- Configurable initial value, drift, and volatility parameters

Usage
-----

The package can be used either through its command-line interface or as a Python module.

Command Line Interface
~~~~~~~~~~~~~~~~~~~~

The simulator can be run directly from the command line using the ``simulate`` command:

.. code-block:: bash

   simulate --y0 1.0 --mu 0.05 --sigma 0.2 --t 0.1 --n 1000 --output gbm_plot.png

Parameters:
   - ``--y0``: Initial value (default: 1.0)
   - ``--mu``: Drift coefficient (default: 0.05)
   - ``--sigma``: Volatility coefficient (default: 0.2)
   - ``--t``: Time duration (default: 0.1)
   - ``--n``: Number of time steps (default: 1000)
   - ``--output``: Output filename for the plot (default: "gbm_plot.png")

Python Module Usage
~~~~~~~~~~~~~~~~~

You can also use the simulator in your Python scripts:

.. code-block:: python

   from C1.gbm_simulator import GBMSimulator

   # Initialize simulator
   simulator = GBMSimulator(y_0=1.0, mu=0.05, sigma=0.2)
   
   # Simulate path
   t_steps, path = simulator.simulate_path(t=0.1, n=1000)

Class Reference
-------------

GBMSimulator
~~~~~~~~~~~

.. autoclass:: C1.gbm_simulator.GBMSimulator
   :members:
   :undoc-members:
   :show-inheritance:

Dependencies
-----------

The package requires Python 3.8 or later and the following main dependencies:

- numpy
- matplotlib
- loguru

For a complete list of dependencies, see the ``pyproject.toml`` file.

License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.