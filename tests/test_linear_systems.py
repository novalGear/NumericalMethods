import tempfile
import unittest
from pathlib import Path

import numpy as np

from LinearSystems.experiment import ExperimentConfig, run_experiment
from LinearSystems.model import LinearSystem, SolverStatus, StoppingCriteria
from LinearSystems.plotting import save_experiment_plots
from LinearSystems.serialization import write_experiment
from LinearSystems.solvers import (
    solve_bicgstab,
    solve_cg,
    solve_gauss_seidel,
    solve_gaussian,
    solve_jacobi,
    solve_lu,
    solve_minimal_residual,
    solve_sor,
    solve_steepest_descent,
)
from LinearSystems.systems import build_variant_l, diagnose_system


class LinearSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = build_variant_l()

    def test_variant_l_diagnostics(self) -> None:
        diagnostics = diagnose_system(self.system)
        self.assertTrue(diagnostics.symmetric)
        self.assertAlmostEqual(diagnostics.eigenvalue_min, 9.65779712, places=7)
        self.assertAlmostEqual(diagnostics.eigenvalue_max, 11.30012840, places=7)
        self.assertAlmostEqual(diagnostics.condition_number_2, 1.17005237, places=7)

    def test_all_nine_methods_converge(self) -> None:
        methods = (
            solve_gaussian,
            solve_lu,
            solve_jacobi,
            solve_gauss_seidel,
            solve_steepest_descent,
            solve_minimal_residual,
            solve_cg,
            solve_bicgstab,
        )
        results = [method(self.system) for method in methods]
        results.append(solve_sor(self.system, 1.01))
        self.assertTrue(
            all(result.status is SolverStatus.CONVERGED for result in results)
        )

    def test_gaussian_pivots_and_inputs_remain_unchanged(self) -> None:
        system = LinearSystem(
            matrix=np.array([[0.0, 1.0], [1.0, 1.0]]),
            rhs=np.array([1.0, 2.0]),
            variant="pivot-test",
        )
        matrix_before = system.matrix.copy()
        rhs_before = system.rhs.copy()
        result = solve_gaussian(system)
        self.assertEqual(result.status, SolverStatus.CONVERGED)
        np.testing.assert_allclose(result.solution, [1.0, 1.0])
        np.testing.assert_array_equal(system.matrix, matrix_before)
        np.testing.assert_array_equal(system.rhs, rhs_before)

    def test_direct_solvers_accept_benign_dynamic_range(self) -> None:
        system = LinearSystem(
            matrix=np.diag([1e20, 1.0]),
            rhs=np.array([1e20, 1.0]),
            variant="scaled-diagonal",
        )
        for result in (solve_gaussian(system), solve_lu(system)):
            self.assertEqual(result.status, SolverStatus.CONVERGED)
            np.testing.assert_allclose(result.solution, [1.0, 1.0])

    def test_jacobi_uses_old_vector_and_seidel_uses_new_components(self) -> None:
        system = LinearSystem(
            matrix=np.array([[4.0, 1.0], [2.0, 3.0]]),
            rhs=np.array([1.0, 2.0]),
            variant="update-order",
        )
        criteria = StoppingCriteria(rtol=0.0, atol=0.0, max_iter=1)
        jacobi = solve_jacobi(system, criteria=criteria)
        seidel = solve_gauss_seidel(system, criteria=criteria)
        np.testing.assert_allclose(jacobi.solution, [0.25, 2.0 / 3.0])
        np.testing.assert_allclose(seidel.solution, [0.25, 0.5])

    def test_sor_at_one_matches_gauss_seidel(self) -> None:
        seidel = solve_gauss_seidel(self.system)
        sor = solve_sor(self.system, 1.0)
        np.testing.assert_array_equal(seidel.solution, sor.solution)
        self.assertEqual(
            [point.residual_abs for point in seidel.history],
            [point.residual_abs for point in sor.history],
        )

    def test_exact_initial_approximation_stops_at_zero(self) -> None:
        exact = np.linalg.solve(self.system.matrix, self.system.rhs)
        result = solve_cg(self.system, exact)
        self.assertEqual(result.status, SolverStatus.CONVERGED)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.history), 1)

    def test_final_history_contains_true_residual(self) -> None:
        for result in (
            solve_jacobi(self.system),
            solve_gauss_seidel(self.system),
            solve_sor(self.system, 1.01),
            solve_steepest_descent(self.system),
            solve_minimal_residual(self.system),
            solve_cg(self.system),
            solve_bicgstab(self.system),
        ):
            expected = np.linalg.norm(
                self.system.rhs - self.system.matrix @ result.solution
            )
            self.assertEqual(result.history[-1].residual_abs, expected)

    def test_bicgstab_breakdown_check_is_scale_invariant(self) -> None:
        matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
        rhs = np.array([1.0, 2.0])
        base = solve_bicgstab(LinearSystem(matrix, rhs, "base"))
        scaled = solve_bicgstab(
            LinearSystem(1e17 * matrix, 1e17 * rhs, "scaled")
        )
        self.assertEqual(base.status, SolverStatus.CONVERGED)
        self.assertEqual(scaled.status, SolverStatus.CONVERGED)
        np.testing.assert_allclose(base.solution, scaled.solution)

    def test_experiment_writes_required_outputs(self) -> None:
        results = run_experiment(
            self.system,
            ExperimentConfig(benchmark_repetitions=1, benchmark_warmups=0),
        )
        self.assertEqual(len(results.metrics), 9)
        self.assertEqual(len(results.sor_scan), 21)
        self.assertEqual(results.best_sor_omegas, (1.0, 1.01))

        with tempfile.TemporaryDirectory() as directory:
            paths = (
                *write_experiment(results, directory),
                *save_experiment_plots(results, directory),
            )
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(Path(path).is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
