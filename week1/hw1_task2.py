"""Task 2: two-user sum-rate problem with a minimum service requirement (slide 36).

    maximize    R = log2(1 + 4*p1) + log2(1 + p2)
    subject to  p1 >= 0,  p2 >= 0,  p1 + p2 <= 1,
                R2 = log2(1 + p2) >= log2(1.25)   (equivalently p2 >= 0.25)
"""

from math import isfinite, log2

import cvxpy as cp
import numpy as np

TOLERANCE = 1e-4
SOLVER = cp.CLARABEL
GAINS = np.array([4.0, 1.0])
R2_MIN = log2(1.25)


def rates(p1, p2):
    """Individual rates (bits/s/Hz), same rate function as task 1."""
    return log2(1 + GAINS[0] * p1), log2(1 + GAINS[1] * p2)


def solve(min_service):
    p = cp.Variable(2, name="p")
    objective = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(GAINS, p))) / np.log(2))
    constraints = {"p >= 0": p >= 0, "p1 + p2 <= 1": cp.sum(p) <= 1}
    if min_service:
        # log2(1 + p2) >= log2(1.25) written as a concave >= constant constraint.
        constraints["R2 >= log2(1.25)"] = cp.log(1 + p[1]) / np.log(2) >= R2_MIN
    problem = cp.Problem(objective, list(constraints.values()))
    problem.solve(solver=SOLVER)
    if p.value is None:
        raise RuntimeError(f"No solution returned; solver status: {problem.status}")
    power1, power2 = (float(v) for v in p.value)
    if not all(isfinite(v) for v in (power1, power2)):
        raise RuntimeError("Solver returned nonfinite results")
    return problem, constraints, power1, power2


def check_constraints(power1, power2, min_service):
    """Constraint slacks computed from the returned numbers, not the solver status."""
    slacks = {
        "p1 >= 0": power1,
        "p2 >= 0": power2,
        "p1 + p2 <= 1": 1 - (power1 + power2),
    }
    if min_service:
        slacks["R2 >= log2(1.25)"] = rates(power1, power2)[1] - R2_MIN
        slacks["p2 >= 0.25"] = power2 - 0.25
    return slacks


def main():
    print(f"CVXPY version: {cp.__version__}")
    results = {}
    for name, min_service in (("Task 1 (no QoS)", False), ("Task 2 (R2 >= log2 1.25)", True)):
        problem, constraints, power1, power2 = solve(min_service)
        slacks = check_constraints(power1, power2, min_service)
        results[name] = (power1, power2)

        print(f"\n=== {name} ===")
        print(f"Solver: {problem.solver_stats.solver_name}, status: {problem.status}")
        print(f"CVXPY objective value: {problem.value:.10f}")
        print("Constraint check from numerical solution (slack >= -tol, violation):")
        feasible = True
        for cname, slack in slacks.items():
            violation = max(0.0, -slack)
            active = abs(slack) < TOLERANCE
            feasible = feasible and violation < TOLERANCE
            print(f"  {cname:<18} slack {slack: .3e}  violation {violation:.3e}"
                  f"  {'ACTIVE' if active else 'inactive'}")
        for cname, con in constraints.items():
            print(f"  dual[{cname}] = {np.round(con.dual_value, 6)}")
        if not feasible:
            raise RuntimeError(f"{name}: returned solution violates a constraint")

    # Comparison table.
    print(f"\n{'Allocation':<26} {'p1':>10} {'p2':>10} {'R1':>10} {'R2':>10} {'Sum rate':>12}")
    sums = {}
    for name, (a, b) in results.items():
        r1, r2 = rates(a, b)
        sums[name] = r1 + r2
        print(f"{name:<26} {a:10.6f} {b:10.6f} {r1:10.6f} {r2:10.6f} {r1 + r2:12.8f}")
    (_, s1), (_, s2) = sums.items()
    print(f"{'Sum-rate loss from QoS':<26} {'':>43} {s1 - s2:12.8f}")

    # Analytical references.
    refs = {"Task 1 (no QoS)": (7 / 8, 1 / 8), "Task 2 (R2 >= log2 1.25)": (3 / 4, 1 / 4)}
    print(f"\nReference checks (absolute error < {TOLERANCE:g}):")
    passed = True
    for name, (e1, e2) in refs.items():
        a, b = results[name]
        errors = (abs(a - e1), abs(b - e2), abs(sums[name] - sum(rates(e1, e2))))
        passed = passed and max(errors) < TOLERANCE
        print(f"  {name:<26} p1 err {errors[0]:.2e}, p2 err {errors[1]:.2e}, "
              f"R err {errors[2]:.2e}")
    print(f"Tolerance check: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main()
