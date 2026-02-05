from scipy.optimize import fsolve
import numpy as np
from sympy import Symbol, solve, symbols

def J(x,y,z):
    return -5*x-4*y-6*z

def g1(x,y,z):
    return x-y+z-20
def g2(x,y,z):
    return 3*x+2*y+4*z-42
def g3(x,y,z):
    return 3*x+2*y-30
def g4(x,y,z):
    return -x
def g5(x,y,z):
    return -y
def g6(x,y,z):
    return -z

def main():
    # From the KKT conditions, we will solve in x, y, z, lambda1...lambda6, alpha1^2...alpha6^2
    # Since this gives us 15 equations and 15 unknowns, we should have a solution (or multiples, since alpha is squared)
    # So we are using scipy fsolve
    solve_numerical()

    # Solve it analytically using sympy
    x, y, z, l1, l2, l3, l4, l5, l6, a1, a2, a3, a4, a5, a6 = symbols("x y z l1 l2 l3 l4 l5 l6 a1 a2 a3 a4 a5 a6")
    analytic_result = solve([-5 + l1 + 3 * l2 + 3 * l3 - l4,
            -4 - l1 + 2 * l2 + 2 * l3 - l5,
            -6 + l1 + 4 * l2 - l6,
            g1(x, y, z) + a1 ** 2,
            g2(x, y, z) + a2 ** 2,
            g3(x, y, z) + a3 ** 2,
            g4(x, y, z) + a4 ** 2,
            g5(x, y, z) + a5 ** 2,
            g6(x, y, z) + a6 ** 2,
            2 * a1 * l1,
            2 * a2 * l2,
            2 * a3 * l3,
            2 * a4 * l4,
            2 * a5 * l5,
            2 * a6 * l6], [x, y, z, l1, l2, l3, l4, l5, l6, a1, a2, a3, a4, a5, a6])

    print("Number of solutions=", len(analytic_result))
    # Go through each solution,
    min_j = 1E6
    num_real_solns = 0
    min_soln = None
    for soln in analytic_result:
        # eliminate any that have complex values.
        if any(val.is_imaginary for val in soln):
            continue
        num_real_solns += 1
        # TODO - Eliminate any that violate the constraints
        # Find the lowest value
        x0,y0,z0 = soln[0], soln[1], soln[2]
        j_val = J(x0,y0,z0).evalf()
        if j_val < min_j:
            min_j = j_val
            min_soln = soln
    # NOTE - Some of these have complex conjugate forms, we ignore that for convenience.
    print("Number real-valued solutions: ", num_real_solns)
    print("Minimum J value: ", min_j)
    print("Minimum J value solution: ", min_soln)
    print("Minimum J value solution x,y,z: ", min_soln[0],min_soln[1],min_soln[2])
    # NOTE - Since some of the alpha values are 0, that means those constraints are satisfied _on the boundary_


def solve_numerical():
    def equations(p):
        x, y, z, l1, l2, l3, l4, l5, l6, a1, a2, a3, a4, a5, a6 = p
        return (
            -5 + l1 + 3 * l2 + 3 * l3 - l4,
            -4 - l1 + 2 * l2 + 2 * l3 - l5,
            -6 + l1 + 4 * l2 - l6,
            g1(x, y, z) + a1 ** 2,
            g2(x, y, z) + a2 ** 2,
            g3(x, y, z) + a3 ** 2,
            g4(x, y, z) + a4 ** 2,
            g5(x, y, z) + a5 ** 2,
            g6(x, y, z) + a6 ** 2,
            2 * a1 * l1,
            2 * a2 * l2,
            2 * a3 * l3,
            2 * a4 * l4,
            2 * a5 * l5,
            2 * a6 * l6
        )

    x0 = 10 * np.ones(15)
    solns = fsolve(equations, x0=x0)
    print(solns)
    # NOTE - fsolve doesn't find all solutions!


if __name__ == "__main__":
    main()
