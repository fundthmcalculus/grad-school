"""Cython-accelerated embedded Runge-Kutta ODE kernels + an exponential integrator.

    from ode_kernels import ode45, odeexp
    result = ode45(lambda t, y: -y, (0.0, 5.0), [1.0])

Six explicit embedded Runge-Kutta pairs, increasing in order, all driven by
one generic adaptive step-size controller styled on
``scipy.integrate.solve_ivp``:

=======  ============================  =====  ===========
method   pair                          order  error order
=======  ============================  =====  ===========
ode12    Heun-Euler                    2      1
ode23    Bogacki-Shampine              3      2   (scipy RK23's own tableau)
ode45    Dormand-Prince                5      4   (scipy RK45's own tableau)
ode56    Verner 6(5) ("Vern6")         6      5
ode67    Verner 7(6) ("Vern7")         7      6
ode78    Fehlberg 7(8) ("RKF78")       8      7
=======  ============================  =====  ===========

Plus ``odeexp``, an exponential Rosenbrock-Euler integrator for mildly
nonlinear stiff systems, with step-doubling adaptivity.

See ``tableaus.py`` for coefficient sources/citations and
``README.md`` for the numerical/implementation notes.
"""

from . import tableaus
from ._common import DenseOutput, OdeResult
from ._expdriver import odeexp
from ._methods import ode12, ode23, ode45, ode56, ode67, ode78

__all__ = [
    "ode12",
    "ode23",
    "ode45",
    "ode56",
    "ode67",
    "ode78",
    "odeexp",
    "OdeResult",
    "DenseOutput",
    "tableaus",
]
