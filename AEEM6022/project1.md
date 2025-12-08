# Project 1
### P1
Even as computers have gotten faster and faster, the need for provably optimal control conditions remains. Numerical techniques have their place. In many cases, numerical techniques are the only techniques that can solve a given problem. In less complicated situations, it is possible to compute the optimal, or near optimal, control inputs required to achieve a desired output position.

For an example problem, consider the thermostat. There is a range of acceptable variance for the temperature from the set point. The tighter this range, the more control input that will be necessary. The variables are: rate of heat loss to the surroundings, rate of heat added by the furnace, and rate of fuel consumed. Assuming a single stage furnace for the purpose of simplicity results in one independent variable. Time. The goal is minimizing the amount of time the furnace spends running.

A more complex version of the same problem would be to have a multi-stage furnace. One that can come on at discrete levels of power. In this case, the goal would be to minimize the total energy consumed over a specific time interval. If the rate of heat loss into the environment is high enough, there would be no solutions, the furnace would run continually.

**Variables:** $Q_{furnace} \gt 0$, $Q_{loss} \lt 0$, $T_{min} \leq T_{set} \leq T_{max}$

**Fitness Function:** $J = \int_{0}^{T} Q_{furnace} dt$

**Constraints:**
1. $T_{min}$ and $T_{max}$ should be close to the set point.
2. $Q_{loss}$ should be less than the heat output rate $Q_{furnace}$ of the furnace.
3. The furnace should not run continuously, nor short-cycle.
### P2-3
The formulation was fairly simple, albeit lengthy. For this reason, `sympy` was used to accelerate the process. The 
setup of the equations was broadly similar to the previous assignments. In this particular case, all the equations were
inequality constraints. Here is the setup:
$$
J = -5x-4y-6z
$$
$$
\theta_1 = x-y+z-20 \leq 0
$$
$$
\theta_2 = 3x+2y+4z-42 \leq 0
$$
$$
\theta_3 = 3x+2y-30 \leq 0
$$
$$
\theta_4 = -x \leq 0
$$
$$
\theta_5 = -y \leq 0
$$
$$
\theta_6 = -z \leq 0
$$

Because of $\theta_{4..6}$, the solution set is the nonnegative subset of ${x,y,z} \in \mathbb{R}^3$. To find the solution, the augmented cost function was augmented with slack variables:
$$
J_{aug} = -5x-4y-6z + \sum_{i=1}^6 \lambda_i [g_i(x,y,z) + \alpha_i^2]
$$

From there, the augmented cost function was differentiated with respect to all variables:
$$
\frac{\partial J_{aug}}{\partial x} = -5 +\lambda_1+3\lambda_2+3\lambda_3-\lambda_4 = 0
$$
$$
\frac{\partial J_{aug}}{\partial y} = -4 -\lambda_1+2\lambda_2+2\lambda_3-\lambda_5 = 0
$$
$$
\frac{\partial J_{aug}}{\partial z} = -6 +\lambda_1+4\lambda_2-\lambda_6 = 0
$$
And so on.


## P4 - 5
A numerical technique was used first, but proved difficult due to the large number of possible solutions. The slack
variables meant that there would be many possible solutions, especially in conjugate (or complex-conjugate) pairs.For this reason, 
For this reason, `sympy` was used to identify all 144 possible solutions at once. The complex solutions were then discarded, leaving 48 real-valued solutions. These 48 solutions were tried in sequence, and the lowest value solution was chosen. This solution was: `x=0`, `y=15`, `z=3`, `J=-78`. 

```
Number of solutions: 144
Number real-valued solutions:  48
Minimum J value:  -78.0000000000000
Minimum J value solution:  (0, 15, 3, 0, 3/2, 1/2, 1, 0, 0, -4*sqrt(2), 0, 0, 0, -sqrt(15), -sqrt(3))
Minimum J value solution x,y,z:  0 15 3
```

KKT necessary conditions are helpful for identifying analytical solutions - the number of constraints can rapidly make the algebraic manipulation difficult or outright impossible. Leveraging Compute Algebra Systems (CAS)s can help with this situation. In other cases, it would probably be easier to guess-and-check the solutions, and then rely on the numerical `fsolve` method to minimize the error.