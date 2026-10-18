"""Solve the two-user sum-rate problem (lecture slide 35) with CVXPY.

    maximize    R = log2(1 + 4*p1) + log2(1 + p2)
    subject to  p1 >= 0,  p2 >= 0,  p1 + p2 <= 1
"""

from math import isfinite, log2

import cvxpy as cp
import numpy as np

TOLERANCE = 1e-4
SOLVER = cp.CLARABEL


def sum_rate(p1, p2):
    """Use the same rate function for optimized and baseline allocations."""
    return log2(1 + 4 * p1) + log2(1 + p2)


def main():
    p = cp.Variable(2, name="p")
    gains = np.array([4.0, 1.0])
    # log2(x) = log(x) / log(2)
    objective = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(gains, p))) / np.log(2))
    constraints = [p >= 0, cp.sum(p) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=SOLVER)

    print(f"CVXPY version: {cp.__version__}")
    print(f"Solver: {problem.solver_stats.solver_name}")
    print(f"Solver status: {problem.status}")
    print(f"Solve time: {problem.solver_stats.solve_time:.6f} seconds")
    print(f"CVXPY objective value: {problem.value:.10f}")
    if p.value is None:
        raise RuntimeError(f"No solution returned; solver status: {problem.status}")
    power1, power2 = (float(v) for v in p.value)
    rate = sum_rate(power1, power2)
    if not all(isfinite(value) for value in (power1, power2, rate)):
        raise RuntimeError("Solver returned nonfinite results")

    print(f"\n{'Allocation':<24} {'p1':>12} {'p2':>12} {'Sum rate':>14}")
    for name, a, b in (
        ("Optimized", power1, power2),
        ("Equal power", 0.5, 0.5),
        ("All power to user 1", 1.0, 0.0),
    ):
        print(f"{name:<24} {a:12.8f} {b:12.8f} {sum_rate(a, b):14.10f}")

    violations = {
        "p1 >= 0": max(0.0, -power1),
        "p2 >= 0": max(0.0, -power2),
        "p1 + p2 <= 1": max(0.0, power1 + power2 - 1),
    }
    print("\nNumerical constraint violations:")
    for name, violation in violations.items():
        print(f"  {name}: {violation:.3e}")
    print(f"  Dual of p1 + p2 <= 1 (bits per unit power): {constraints[1].dual_value:.8f}")

    # Water-filling: p_i = (mu - 1/g_i)^+ with sum p_i = 1.
    # 1/4 + p1 = 1 + p2 and p1 + p2 = 1  ->  p1 = 7/8, p2 = 1/8.
    references = (
        ("p1", power1, 7 / 8),
        ("p2", power2, 1 / 8),
        ("Optimal sum rate", rate, sum_rate(7 / 8, 1 / 8)),
    )
    print(f"\nReference checks (absolute error < {TOLERANCE:g}):")
    passed = True
    for name, actual, expected in references:
        error = abs(actual - expected)
        passed = passed and error < TOLERANCE
        print(f"  {name}: expected {expected:.10f}, error {error:.3e}")
    passed = passed and max(violations.values()) < TOLERANCE
    print(f"Tolerance check: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError("Solution exceeds the required absolute tolerance")


if __name__ == "__main__":
    main()
