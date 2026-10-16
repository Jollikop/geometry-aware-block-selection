from __future__ import annotations

import ast
import json
import os
import gzip
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SIM_ROOT.parent
SVGP_ROOT = SIM_ROOT / "svgp"

from common import datasets  # noqa: E402
from common import stats  # noqa: E402
from common.provenance import (  # noqa: E402
    RELEASE_PACKAGE_VERSIONS,
    SVGP_RELEASE_PACKAGES,
    code_state,
    config_fingerprint,
    release_runtime_matches,
    sha256_file,
)
from sgpr import benchmark as jax_sgpr_bench  # noqa: E402
from sgpr import initialization as sgpr_initialization  # noqa: E402
from sgpr import jax_collapsed as jax_models  # noqa: E402
from sgpr import protocol as sgpr_protocol  # noqa: E402
from svgp import plans  # noqa: E402
from svgp import plot_inducing_curves as inducing  # noqa: E402


class SubmissionProtocolTests(unittest.TestCase):
    def test_sha256_is_content_based(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.bin"
            path.write_bytes(b"selected-blocks")
            first = sha256_file(path)
            path.write_bytes(b"selected-blocks!")
            self.assertTrue(first.startswith("sha256:"))
            self.assertNotEqual(first, sha256_file(path))

    def test_config_fingerprint_is_canonical(self) -> None:
        self.assertEqual(
            config_fingerprint({"b": 2, "a": 1}),
            config_fingerprint({"a": 1, "b": 2}),
        )
        self.assertNotEqual(
            config_fingerprint({"a": 1}),
            config_fingerprint({"a": 2}),
        )

    def test_release_runtime_profile_is_exact(self) -> None:
        runtime = {
            "python": "3.12.9",
            "packages": {
                name: RELEASE_PACKAGE_VERSIONS[name]
                for name in SVGP_RELEASE_PACKAGES
            },
            "jax_devices": [{"platform": "cpu", "id": 0}],
        }
        self.assertTrue(release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES))
        runtime["packages"]["optax"] = "0.0.0"
        self.assertFalse(release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES))

        runtime["packages"]["optax"] = RELEASE_PACKAGE_VERSIONS["optax"]
        runtime["jax_devices"] = [{"platform": "gpu", "id": 0}]
        # ptxas joined the GPU profile in v10: jax[cuda12] leaves it unpinned,
        # so two hosts with identical jax versions can compile different device
        # code. It is checked only when the run is on a GPU, which is why a CPU
        # runtime above passes without it.
        runtime["packages"].update({
            name: RELEASE_PACKAGE_VERSIONS[name]
            for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt",
                         "nvidia-cuda-nvcc-cu12")
        })
        runtime["nvidia_smi"] = ["NVIDIA H100 PCIe, 550.54.15"]
        self.assertTrue(release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES))
        runtime["packages"].pop("nvidia-cuda-nvcc-cu12")
        self.assertFalse(release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES))
        runtime["packages"]["nvidia-cuda-nvcc-cu12"] = (
            RELEASE_PACKAGE_VERSIONS["nvidia-cuda-nvcc-cu12"]
        )
        runtime["nvidia_smi"] = []
        self.assertFalse(release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES))

    def test_code_state_is_scoped_and_includes_untracked_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            simulations = root / "Simulations"
            simulations.mkdir()
            (simulations / "runner.py").write_text("VALUE = 1\n")
            (root / "paper.tex").write_text("draft\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "ci@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "CI"],
                check=True,
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"],
                check=True,
            )

            (root / "paper.tex").write_text("unrelated draft\n")
            code_state.cache_clear()
            self.assertFalse(code_state(root)["source_dirty"])

            (simulations / "new_runner.py").write_text("VALUE = 2\n")
            code_state.cache_clear()
            state = code_state(root)
            self.assertTrue(state["source_dirty"])
            self.assertTrue(any("new_runner.py" in row for row in state["dirty_entries"]))

    def test_every_figure_uses_the_shared_error_bar(self) -> None:
        # Three scripts used to carry their own ``uncertainty``. The contract
        # itself is pinned in tests/test_stats.py; what matters here is that no
        # figure defines a private one again.
        for script in ("svgp/plot_inducing_curves.py",
                       "svgp/plot_batch_separation_ablation.py",
                       "sgpr/plot_predictive_nll_ablation.py"):
            source = (SIM_ROOT / script).read_text()
            self.assertNotIn("def uncertainty(", source, script)
            self.assertIn("st.uncertainty(", source, script)
        self.assertIs(inducing.st.uncertainty, stats.uncertainty)

    def test_aot_cache_signature_distinguishes_state_dtypes(self) -> None:
        state64 = {"num_linesearch_steps": np.asarray(0, dtype=np.int64)}
        state32 = {"num_linesearch_steps": np.asarray(0, dtype=np.int32)}
        self.assertNotEqual(
            jax_models._pytree_abstract_signature(state64),
            jax_models._pytree_abstract_signature(state32),
        )

    def test_metis_weights_use_flat_csr_layout(self) -> None:
        edge_maps = [
            {2: 5, 1: 3},
            {0: 3},
            {0: 5},
        ]
        xadj, adjncy, eweights = jax_sgpr_bench._metis_weighted_csr(edge_maps)
        self.assertEqual(xadj, [0, 2, 3, 4])
        self.assertEqual(adjncy, [1, 2, 0, 0])
        self.assertEqual(eweights, [3, 5, 3, 5])

    def test_cache_requires_matching_source_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            datasets._write_cache("pol", root, [[0.0], [1.0]], [0.0, 1.0])
            x, y = datasets._load_verified_cache("pol", root)
            self.assertEqual(x.shape, (2, 1))
            self.assertEqual(y.shape, (2,))

            sidecar = datasets.provenance_path("pol", root)
            metadata = json.loads(sidecar.read_text())
            metadata["source"] = "wrong-source"
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                datasets._load_verified_cache("pol", root)

    def test_traffic_loader_accepts_compressed_csv_member(self) -> None:
        csv = (
            b"date_time,traffic_volume\n"
            b"2012-10-02 09:00:00,5545\n"
            b"2012-10-02 10:00:00,4516\n"
        )
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "Metro_Interstate_Traffic_Volume.csv.gz",
                gzip.compress(csv),
            )
        with mock.patch.object(
            datasets.urllib.request,
            "urlopen",
            return_value=io.BytesIO(payload.getvalue()),
        ):
            x, y = datasets._load_traffic()
        self.assertEqual(x.shape, (2, 3))
        np.testing.assert_array_equal(y, [5545.0, 4516.0])

    def test_force_reaches_custom_dataset_loader(self) -> None:
        loader = mock.Mock(
            return_value=(
                np.asarray([[0.0], [1.0]]),
                np.asarray([0.0, 1.0]),
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(
                datasets._CUSTOM_LOADERS,
                {"superconductivity": loader},
            ):
                datasets.load_raw("superconductivity", folder, force=True)
        loader.assert_called_once_with(force=True)

    def test_current_ids_and_exact_method_registries(self) -> None:
        sgpr = (SIM_ROOT / "sgpr/benchmark.py").read_text()
        svgp = (SIM_ROOT / "svgp/engine.py").read_text()
        self.assertIn("sgpr-collapsed-sbt-jax-v23", sgpr)
        self.assertIn("svgp-residual-gain-double-batch-jax-v12", svgp)
        sgpr_tree = ast.parse(sgpr)
        svgp_tree = ast.parse(svgp)
        sgpr_methods = next(
            ast.literal_eval(node.value)
            for node in sgpr_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "METHODS"
                    for target in node.targets)
        )
        svgp_methods = next(
            ast.literal_eval(node.value)
            for node in svgp_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PAPER_METHODS"
                    for target in node.targets)
        )
        self.assertEqual(set(sgpr_methods), {
            "Exact-GP", "E-SGPR", "METIS-SGPR", "SGPR", "T-SGPR",
            "RBT-SGPR", "BT-SGPR", "SBT-SGPR",
        })
        self.assertEqual(set(svgp_methods), {
            "SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP",
        })

    def test_sgpr_release_requires_scipy_convergence(self) -> None:
        self.assertEqual(
            sgpr_protocol.LBFGS_REFRESH_SCHEDULE,
            (15, 15, 20, 25, 99925),
        )
        self.assertEqual(sgpr_protocol.LBFGS_STEPS, 100000)
        self.assertEqual(sgpr_protocol.CONVERGENCE_WINDOW_STEPS, 100)
        self.assertEqual(sgpr_protocol.CONVERGENCE_OBJ_N_TOL, 1e-4)
        self.assertEqual(sgpr_protocol.CONVERGENCE_GRAD_MAX_N_TOL, 1e-3)
        self.assertEqual(sgpr_protocol.SCIPY_LBFGSB_MAXCOR, 100)
        self.assertEqual(sgpr_protocol.SCIPY_LBFGSB_MAXFUN, 250000)
        launcher = (SIM_ROOT / "sgpr/paper_sgpr_main.sbatch").read_text()
        self.assertIn("15,15,20,25,99925", launcher)
        self.assertIn("--require-convergence", launcher)
        self.assertIn('--logs-dir "$LOGS_DIR"', launcher)
        # The v23 release campaign itself was launched by this worker.
        release = (SIM_ROOT / "sgpr_full_worker.sbatch").read_text()
        self.assertIn("15,15,20,25,99925", release)
        self.assertIn("--require-convergence", release)
        self.assertIn("scipy-l-bfgs-b", jax_models._train_scipy_lbfgs.__doc__)

        stable = jax_sgpr_bench.objective_convergence_summary(
            np.asarray([[0, 20.0], [900, 10.0005], [999, 10.0001]]),
            final_loss=10.0,
            n_points=10,
            total_steps=1000,
        )
        self.assertTrue(stable["convergence_passed"])

        drifting = jax_sgpr_bench.objective_convergence_summary(
            np.asarray([[0, 20.0], [900, 10.002], [999, 10.001]]),
            final_loss=10.0,
            n_points=10,
            total_steps=1000,
        )
        self.assertFalse(drifting["convergence_passed"])

    def test_sgpr_release_initializations_are_nested_and_deterministic(self) -> None:
        x = jnp.asarray(np.random.default_rng(4).normal(size=(24, 3)))
        params = jax_models.init_params(x, x[:8], kernel_name="rbf")
        for strategy in ("greedy-variance", "kmeans++"):
            first = sgpr_initialization.inducing_indices(
                x,
                3,
                params,
                kernel_name="rbf",
                seed=7,
                strategy=strategy,
                nested_max_m=8,
            )
            second = sgpr_initialization.inducing_indices(
                x,
                5,
                params,
                kernel_name="rbf",
                seed=7,
                strategy=strategy,
                nested_max_m=8,
            )
            repeated = sgpr_initialization.inducing_indices(
                x,
                5,
                params,
                kernel_name="rbf",
                seed=7,
                strategy=strategy,
                nested_max_m=8,
            )
            np.testing.assert_array_equal(first, second[:3])
            np.testing.assert_array_equal(second, repeated)
            self.assertEqual(len(set(map(int, second))), 5)

    def test_full_campaign_uses_exactly_five_long_lived_workers(self) -> None:
        worker = (SIM_ROOT / "paper_five_worker.sbatch").read_text()
        submitter = (SIM_ROOT / "submit_five_worker_campaign.sh").read_text()
        self.assertIn("#SBATCH --array=0-4%5", worker)
        self.assertIn("#SBATCH --no-requeue", worker)
        self.assertIn("export GPU_MAX_REQUEUE=0", worker)
        self.assertNotIn("sbatch ", worker)
        self.assertIn("--array=0-4%5", submitter)
        self.assertEqual(submitter.count("CAMPAIGN_JOB=$(sbatch "), 1)
        self.assertIn("FIVE_WORKER_FINAL_ROOT", submitter)
        self.assertNotIn('FINAL_ROOT="${FINAL_ROOT:-', submitter)
        for stage in (
            "paper_sgpr_main.sbatch",
            "paper_sgpr_transplant.sbatch",
            "paper_grid_stationary.sbatch",
            "paper_grid_gibbs.sbatch",
            "paper_sparse_transplant.sbatch",
            "paper_convergence.sbatch",
        ):
            self.assertIn(stage, worker)
        self.assertIn("waiting-for-all-svgp-donors", worker)

    def test_sgpr_sweep_preserves_the_running_svgp_campaign(self) -> None:
        worker = (SIM_ROOT / "sgpr_sweep_worker.sbatch").read_text()
        submitter = (SIM_ROOT / "submit_sgpr_sweep_campaign.sh").read_text()
        watcher = (SIM_ROOT / "watch_sgpr_sweep_campaign.sh").read_text()
        self.assertIn("#SBATCH --array=0-4%1", worker)
        self.assertIn("#SBATCH --no-requeue", worker)
        self.assertNotIn("sbatch ", worker)
        self.assertEqual(submitter.count("SWEEP_JOB=$(sbatch "), 1)
        self.assertIn("PRESERVED_CAMPAIGN_ROOT", submitter)
        self.assertNotIn("scancel", submitter)
        self.assertNotIn("/svgp/", worker)
        for value in ("128", "256", "512"):
            self.assertIn(value, watcher)
        self.assertIn("PRESERVED SVGP", watcher)

    def test_svgp_resume_preserves_historical_fingerprints(self) -> None:
        worker = (SIM_ROOT / "svgp_resume_worker.sbatch").read_text()
        bundle = (SIM_ROOT / "svgp_resume_bundle.sbatch").read_text()
        stage = (SIM_ROOT / "run_svgp_resume_stage.sh").read_text()
        submitter = (SIM_ROOT / "submit_svgp_scope_campaign.sh").read_text()
        all_submitter = (SIM_ROOT / "submit_svgp_all_campaign.sh").read_text()
        road = (SIM_ROOT / "submit_svgp_3droad_campaign.sh").read_text()
        watcher = (SIM_ROOT / "watch_svgp_resume_campaign.sh").read_text()
        self.assertIn("SOURCE_PROJECT_ROOT", worker)
        self.assertIn("EXPECTED_SOURCE_HEAD", worker)
        self.assertIn("XLA_PYTHON_CLIENT_ALLOCATOR=platform", worker)
        self.assertIn("RESUME_WORKER_ID", worker)
        self.assertIn("--exclusive", bundle)
        self.assertIn("--gres=gpu:1", bundle)
        self.assertIn("svgp_resume_runner.py", stage)
        self.assertIn("--prediction-chunk", stage)
        self.assertIn(
            "kernel=$kernel dataset=$dataset method=$method", stage
        )
        self.assertIn("--plan grid", stage)
        self.assertIn("--plan grid-exploratory", stage)
        self.assertIn("--plan custom", stage)
        self.assertIn('--seeds "$seed"', stage)
        self.assertIn('seed<10; seed+=NSHARDS', stage)
        self.assertNotIn("--force", stage)
        self.assertNotIn("scancel", submitter)
        self.assertIn('$(<"$FINAL_ROOT/EXPECTED_HEAD")', submitter)
        self.assertIn('RESUME_DATASETS="3droad"', submitter)
        self.assertIn(
            'RESUME_DATASETS="pol superconductivity kin40k protein elevators traffic electricity 3droad"',
            submitter,
        )
        self.assertIn('submit_svgp_scope_campaign.sh" all', all_submitter)
        self.assertIn('NSHARDS=$CONCURRENCY', submitter)
        self.assertIn('0-$((CONCURRENCY - 1))%$CONCURRENCY', submitter)
        self.assertIn('EXECUTION_MODE', submitter)
        self.assertIn('svgp_resume_bundle.sbatch', submitter)
        self.assertIn('--dependency="$SVGP_RESUME_DEPENDENCY"', submitter)
        self.assertIn('SVGP_RESUME_STAGES', submitter)
        self.assertIn('SVGP_RESUME_EXTENDED_KERNELS', stage)
        self.assertIn('SVGP_RESUME_TRANSPLANT_KERNELS', stage)
        self.assertIn('SVGP_RESUME_PREDICTION_MODE', stage)
        self.assertIn('--defer-prediction', stage)
        self.assertIn('--prediction-only', stage)
        self.assertIn('aistats2026_svgp_gibbs_rbf_warm300_v13', stage)
        self.assertIn('--init-from-kernel rbf', stage)
        self.assertIn('--epochs 300', stage)
        self.assertIn('--refresh-until 81', stage)
        self.assertIn(
            '"init_params_sha256"',
            (SVGP_ROOT / "run_benchmark.py").read_text(),
        )
        self.assertIn('stage_enabled transplant', worker)
        self.assertIn('RESUME_EXECUTION_MODE', watcher)
        self.assertIn('ACTIVE TARGETS', watcher)
        self.assertIn('transplant predicted', watcher)
        self.assertIn('SVGP_NUMERICAL_REFERENCE_HEAD', submitter)
        self.assertIn('NUMERICAL_CHANGES', submitter)
        self.assertIn('Simulations/common Simulations/svgp', submitter)
        self.assertIn("\\.(batch|extern)", watcher)
        self.assertIn('task<4; task+=NSHARDS', worker)
        self.assertIn('set -- "$@" 1', road)

    def test_residual_figure_parser_uses_the_release_protocol(self) -> None:
        from sgpr.residual_illustration import (
            run_residual_structure_kin40k as residual,
        )

        with mock.patch.object(sys, "argv", ["residual-figure"]):
            args = residual.parse_args()
        self.assertEqual(args.steps, sgpr_protocol.LBFGS_STEPS)
        self.assertEqual(
            args.refresh_schedule,
            ",".join(map(str, sgpr_protocol.LBFGS_REFRESH_SCHEDULE)),
        )

    def test_lbfgs_convergence_stops_only_in_final_fixed_phase(self) -> None:
        def quadratic(params, *_terms):
            return jnp.sum(params["z"] ** 2)

        dummy = (
            jnp.zeros((1,)),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.zeros((1, 1)),
        )
        schedule = [1, 1, 1, 1, 40]
        _, _, info = jax_models.train_collapsed(
            quadratic,
            {"z": jnp.asarray([1.0])},
            terms0=dummy,
            optimizer="lbfgs",
            refresh_schedule=schedule,
            convergence_abs_tol=1e-12,
            convergence_window_steps=5,
            convergence_patience=2,
            convergence_grad_max_tol=1e-8,
        )
        self.assertTrue(info["convergence_reached"])
        self.assertEqual(info["optimizer_backend"], "scipy-l-bfgs-b")
        self.assertLessEqual(info["actual_steps"], sum(schedule))
        self.assertEqual(info["refreshes"], 0)
        self.assertEqual(len(info["phase_actual_iters"]), 5)
        self.assertTrue(info["optimizer_terminal_convergence"])
        self.assertTrue(info["convergence_checks"][-1]["solver_terminal"])
        self.assertLessEqual(info["convergence_checks"][-1]["budget"], 5)

        _, _, capped = jax_models.train_collapsed(
            quadratic,
            {"z": jnp.asarray([1.0])},
            terms0=dummy,
            optimizer="lbfgs",
            refresh_schedule=schedule,
            convergence_abs_tol=1e-12,
            convergence_window_steps=5,
            convergence_patience=100,
            convergence_grad_max_tol=-1.0,
        )
        self.assertFalse(capped["convergence_reached"])
        self.assertLessEqual(capped["actual_steps"], sum(schedule))

        unaligned = [1, 1, 1, 1, 12]
        _, _, aligned = jax_models.train_collapsed(
            quadratic,
            {"z": jnp.asarray([1.0])},
            terms0=dummy,
            optimizer="lbfgs",
            refresh_schedule=unaligned,
            convergence_abs_tol=1e-12,
            convergence_window_steps=5,
            convergence_patience=100,
            convergence_grad_max_tol=-1.0,
        )
        self.assertLessEqual(aligned["actual_steps"], sum(unaligned))
        self.assertFalse(aligned["convergence_reached"])

        def rosenbrock(params, *_terms):
            z = params["z"]
            return jnp.sum(
                100.0 * (z[1:] - z[:-1] ** 2) ** 2
                + (1.0 - z[:-1]) ** 2
            )

        long_schedule = [1, 1, 1, 1, 200]
        _, _, continuous = jax_models.train_collapsed(
            rosenbrock,
            {"z": jnp.asarray([-1.2, 1.0])},
            terms0=dummy,
            optimizer="lbfgs",
            refresh_schedule=long_schedule,
            convergence_abs_tol=1e-2,
            convergence_window_steps=5,
            convergence_patience=2,
            convergence_grad_max_tol=1e-1,
        )
        self.assertTrue(continuous["convergence_reached"])
        self.assertFalse(continuous["optimizer_terminal_convergence"])
        self.assertLess(continuous["actual_steps"], sum(long_schedule))
        self.assertEqual(len(continuous["scipy_attempts"]), 5)
        self.assertEqual(continuous["scipy_attempts"][-1]["maxiter"], 200)
        self.assertEqual(continuous["scipy_attempts"][-1]["status"], 99)
        self.assertTrue(all(
            check["stable"] and check["budget"] == 5
            for check in continuous["convergence_checks"][-2:]
        ))

    def test_snelson_separates_exact_and_full_residual_models(self) -> None:
        source = (SIM_ROOT / "sgpr/snelson/run_snelson.py").read_text()
        self.assertIn("true_gp = gpflow.models.GPR", source)
        self.assertIn(
            "exact_sgpr = ensure_trainable_z(FullResidualSGPR", source
        )

    def test_only_the_exact_gp_uses_the_true_gp_predictor(self) -> None:
        # E-SGPR is a bound, not an exact GP: it keeps the sparse posterior mean
        # and reports Eq. (12), exactly like the other residual-modelling
        # methods. Giving it the true GP predictor once made it look better than
        # its own bound could justify.
        from common import predictors as pr

        self.assertEqual(
            {method for method, predictor in
             pr.COLLAPSED_METHOD_PREDICTOR.items() if predictor == "exact"},
            {"Exact-GP"},
        )
        self.assertEqual(pr.COLLAPSED_METHOD_PREDICTOR["E-SGPR"], "full")
        source = (SIM_ROOT / "sgpr/benchmark.py").read_text()
        self.assertIn('predictor = METHOD_PREDICTOR[method]', source)
        self.assertIn('exact_gp_predictive = predictor == "exact"', source)

    def test_code_state_is_given_the_directory_that_holds_simulations(self) -> None:
        # code_state runs ``git status -- Simulations`` from the root it is
        # handed, so a root one level too deep makes the pathspec match nothing
        # and every run reports source_dirty=False. That is worse than a crash:
        # the manifest then certifies a dirty tree as clean.
        import ast as _ast

        for module in ("sgpr/snelson/run_snelson.py",
                       "sgpr/benchmark.py",
                       "svgp/run_benchmark.py"):
            path = SIM_ROOT / module
            source = path.read_text()
            depth = len(path.relative_to(SIM_ROOT).parts) - 1
            for node in _ast.walk(_ast.parse(source)):
                if not (isinstance(node, _ast.Call)
                        and getattr(node.func, "id", "") == "code_state"):
                    continue
                argument = _ast.get_source_segment(source, node.args[0])
                # The root must be SIM_ROOT's parent: that is ``parents[depth]``
                # counted from the module, or a name bound to it.
                self.assertTrue(
                    f"parents[{depth + 1}]" in argument
                    or argument in {"REPO_ROOT", "PROJECT_ROOT"}
                    or argument.count(".parent") == depth + 1,
                    f"{module}: code_state({argument}) is not the directory "
                    "that contains Simulations/",
                )

    def test_public_kernel_registry_is_exact(self) -> None:
        source = (SIM_ROOT / "common/kernels.py").read_text()
        tree = ast.parse(source)
        kernels = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "KERNEL_ORDER"
                    for target in node.targets)
        )
        self.assertEqual(kernels, [
            "rbf", "amplitude-rbf", "matern32", "matern52", "gibbs",
            "periodic-angular", "matern-periodic",
        ])

    def test_report_registry_is_exactly_the_eight_declared_pairs(self) -> None:
        # v10: bike is retired for target leakage (the target is
        # log(casual + registered) and both columns are features); co2 for
        # being too small to have more than one block at B = 500, which makes
        # the block methods coincide. Both keep a registry entry so archived
        # campaigns stay loadable, and are excluded from every plan by
        # assert_not_retired. elevators replaces bike at matched size.
        self.assertEqual(set(datasets.DATASET_KEYS), {
            "kin40k", "protein", "3droad", "traffic", "electricity",
            "superconductivity", "co2", "bike", "pol", "elevators",
        })
        self.assertEqual(set(datasets.RETIRED_DATASETS), {"bike", "co2"})
        self.assertEqual(plans.build_plan("report"), plans.REPORT_PAIRS)
        self.assertEqual(len(plans.REPORT_PAIRS), 8)
        self.assertEqual(len(plans.GRID_DATASETS), 8)
        self.assertFalse(
            set(plans.GRID_DATASETS) & set(datasets.RETIRED_DATASETS))
        with self.assertRaises(ValueError):
            plans.grid_pairs(["rbf"], ["kin40k", "bike"])
        self.assertEqual(
            {name: datasets.dataset_source(name) for name in datasets.DATASET_KEYS},
            {
                "pol": "OpenML:201",
                "bike": "uci_datasets-1.1.0:bike",
                "elevators": "uci_datasets-1.1.0:elevators",
                "kin40k": "uci_datasets-1.1.0:kin40k",
                "protein": "uci_datasets-1.1.0:protein",
                "3droad": "uci_datasets-1.1.0:3droad",
                "traffic": "UCI:492",
                "co2": "NOAA-GML:MLO-monthly",
                "electricity": "UCI:321:prepared-v2",
                "superconductivity": "UCI:464",
            },
        )

    def test_svgp_fingerprint_covers_release_critical_controls(self) -> None:
        source = (
            SVGP_ROOT / "run_benchmark.py"
        ).read_text()
        required = {
            "batch_coupling", "penalty_sampling", "data_signature",
            "data_source", "record_history", "save_params",
            "full_predictor_max_n", "reported_predictor",
            "split_protocol", "split_seed", "split_rng", "model_seed",
            "preprocessing_protocol",
            "train_indices_sha256", "test_indices_sha256",
            "code_source_dirty",
        }
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_run_config"
        )
        config_literal = next(
            node.value for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "config"
                    for target in node.targets)
            and isinstance(node.value, ast.Dict)
        )
        keys = {
            key.value for key in config_literal.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertTrue(required <= keys, required - keys)

    def test_paper_status_runs_from_project_root(self) -> None:
        # Run it the way a reviewer would: from the repository root, on a
        # checkout that has not been pip-installed. SIM_ROOT on PYTHONPATH is
        # what ``pip install -e Simulations/`` does permanently.
        env = dict(os.environ, PYTHONPATH=str(SIM_ROOT))
        completed = subprocess.run(
            [
                sys.executable,
                str(SVGP_ROOT / "analysis/paper_status.py"),
                "--brief",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("complete=", completed.stdout)
        self.assertIn("release_ready=", completed.stdout)

    def test_the_sgpr_table_names_its_predictor_in_every_row(self) -> None:
        source = (SIM_ROOT / "sgpr/make_main_table.py").read_text()
        tree = ast.parse(source)
        write_tex = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_tex"
        )
        body = ast.get_source_segment(source, write_tex)
        # Method, predictor label, three aggregated metrics, prediction time.
        # ``cell`` renders an already-aggregated summary and is deliberately
        # not common.stats.mean_std, which reduces a raw sample instead.
        self.assertEqual(body.count("cell(values,"), 3)
        self.assertIn("predictor_label[METHOD_PREDICTOR[method]]", body)
        self.assertIn("pred_ms_mean", body)

    def test_no_table_or_figure_mixes_two_predictors_in_one_column(self) -> None:
        # The estimator a row reports is a property of the method, declared once
        # in common.predictors. Any consumer that hard-codes a per-method
        # predictor map is a second source of truth and will drift.
        from common import predictors as pr

        for module in ("sgpr/make_main_table.py",
                       "sgpr/plot_block_convergence.py"):
            source = (SIM_ROOT / module).read_text()
            self.assertNotIn("standard_sparse", source, module)
            self.assertNotIn("training_local_cross", source, module)
        self.assertEqual(
            set(pr.COLLAPSED_METHOD_PREDICTOR.values()) |
            set(pr.VARIATIONAL_METHOD_PREDICTOR.values()),
            {"sparse", "diagonal", "block", "full", "exact"},
        )


if __name__ == "__main__":
    unittest.main()
