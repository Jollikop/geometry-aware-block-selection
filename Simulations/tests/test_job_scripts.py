"""Every cluster job must be launchable against the runner it calls.

A stale flag in an .sbatch costs a queue slot and a night: Slurm accepts the
submission, the job starts, and argparse kills it on the first line. These
tests read what a job passes and check it against its runner's parser, so a
renamed or deleted option fails here instead of on the cluster.

The declaration is exhaustive on purpose. An earlier version listed eight of
the twenty-one jobs and checked the runs root on two of them, so it walked
straight past the one launcher that still wrote to a retired tree.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[1]

#: Every cluster job, and the runners it invokes. Adding a job without adding
#: it here fails ``test_every_job_is_declared``.
JOB_RUNNERS = {
    "analysis/partition_controls_certify.sbatch": ("analysis/partition_controls.py",),
    "analysis/dense_cutoff_certify.sbatch": ("analysis/dense_cutoff_sweep.py",),
    "analysis/dense_cutoff_sweep.sbatch": ("analysis/dense_cutoff_sweep.py",),
    "analysis/batch_ablation_block_diag.sbatch": ("analysis/batch_ablation_block_diag.py",),
    "analysis/hard_block_prediction_campaign.sbatch": ("analysis/hard_block_prediction_campaign.py",),
    "analysis/hard_block_prediction_certify.sbatch": ("analysis/hard_block_prediction_campaign.py",),
    "analysis/weighted_residual_audit.sbatch": ("analysis/weighted_residual_audit.py",),
    "analysis/weighted_residual_campaign.sbatch": ("analysis/weighted_residual_campaign.py",),
    "analysis/partition_controls.sbatch": ("analysis/partition_controls.py",),
    "analysis/partition_controls_first_batch.sbatch": ("analysis/partition_controls.py",),
    "svgp_resume_bundle.sbatch": ("svgp_resume_worker.sbatch",),
    "svgp_resume_worker.sbatch": ("svgp/run_benchmark.py",),
    "sgpr_sweep_worker.sbatch": (
        "sgpr/run_benchmark.py",
        "sgpr/analysis/explore_hyperparameter_transplant.py",
    ),
    "paper_five_worker.sbatch": (
        "sgpr/run_benchmark.py",
        "sgpr/analysis/explore_hyperparameter_transplant.py",
        "sgpr/residual_illustration/run_residual_structure_kin40k.py",
        "svgp/run_benchmark.py",
    ),
    "paper_finalize.sbatch": (
        "analysis/validate_final_suite.py", "analysis/make_final_figures.py",
        "sgpr/analysis/explore_hyperparameter_transplant.py"),
    "sgpr/paper_sgpr_main.sbatch": ("sgpr/run_benchmark.py",),
    "sgpr/paper_sgpr_transplant.sbatch": (
        "sgpr/analysis/explore_hyperparameter_transplant.py",),
    "sgpr/paper_sgpr_finalize.sbatch": (
        "analysis/validate_final_suite.py", "sgpr/make_main_table.py",
        "sgpr/analysis/explore_hyperparameter_transplant.py"),
    "sgpr/predictive_nll_ablation.sbatch": ("sgpr/run_benchmark.py",),
    "sgpr/residual_illustration/paper_residual_kin40k.sbatch": (
        "sgpr/residual_illustration/run_residual_structure_kin40k.py",),
    "sgpr/snelson/paper_snelson.sbatch": ("sgpr/snelson/run_snelson.py",),
    "sgpr/analysis/fixed_theta_bounds.sbatch": (
        "sgpr/analysis/explore_fixed_theta_bounds.py",),
    "sgpr/analysis/fixed_theta_bounds_cpu.sbatch": (
        "sgpr/analysis/explore_fixed_theta_bounds.py",),
    "sgpr/analysis/fixed_theta_bounds_summary.sbatch": (
        "sgpr/analysis/explore_fixed_theta_bounds.py",),
    "sgpr/analysis/hyperparameter_transplant.sbatch": (
        "sgpr/analysis/explore_hyperparameter_transplant.py",),
    "sgpr/analysis/hyperparameter_transplant_summary.sbatch": (
        "sgpr/analysis/explore_hyperparameter_transplant.py",),
    "sgpr/analysis/sbt_guided_sgpr_exploratory.sbatch": (
        "sgpr/analysis/explore_sbt_guided_sgpr.py",),
    "sgpr/analysis/sbt_guided_sgpr_summary.sbatch": (
        "sgpr/analysis/explore_sbt_guided_sgpr.py",),
    "svgp/paper_grid_stationary.sbatch": ("svgp/run_benchmark.py",),
    "svgp/paper_grid_gibbs.sbatch": ("svgp/run_benchmark.py",),
    "svgp/paper_sparse_transplant.sbatch": ("svgp/run_benchmark.py",),
    "svgp/paper_convergence.sbatch": ("svgp/run_benchmark.py",),
    "svgp/paper_sbt_structure_refresh.sbatch": ("svgp/run_benchmark.py",),
    "svgp/paper_report_4methods.sbatch": ("svgp/run_benchmark.py",),
    "svgp/batch_separation_ablation.sbatch": ("svgp/run_benchmark.py",),
    "sgpr_full_worker.sbatch": (
        "sgpr/run_benchmark.py",
        "sgpr/analysis/explore_hyperparameter_transplant.py",
    ),
    "gibbs_anchor_worker.sbatch": ("gibbs_anchor_campaign.py",),
    "gibbs_anchor_prediction_worker.sbatch": (
        "analysis/gibbs_anchor_predictions.py",),
    "svgp_missing_predictions.sbatch": (
        "analysis/run_svgp_missing_predictions.py",),
    "svgp_transplant_continue.sbatch": (
        "analysis/continue_svgp_transplant.py",),
    "gibbs_anchor_transplant_certify.sbatch": (
        "analysis/certify_gibbs_anchor_transplant.py",),
    "gibbs_anchor_transplant_cpu.sbatch": (
        "analysis/gibbs_anchor_transplant_scope.py",
        "analysis/continue_svgp_transplant.py",),
    "gibbs_anchor_transplant_gpu.sbatch": (
        "svgp/run_benchmark.py",
        "analysis/gibbs_anchor_transplant_scope.py",
        "analysis/continue_svgp_transplant.py",
        "analysis/evaluate_svgp_frozen.py",),
    "analysis/real_predictor_main.sbatch": (
        "analysis/real_predictor_campaign.py",),
    "analysis/hard_block_prediction_campaign.sbatch": (
        "analysis/hard_block_prediction_campaign.py",),
    "analysis/hard_block_prediction_certify.sbatch": (
        "analysis/hard_block_prediction_campaign.py",),
    "analysis/batch_ablation_block_diag.sbatch": (
        "analysis/batch_ablation_block_diag.py",),
}

#: Options consumed by sbatch, git or nvidia-smi rather than by the runner.
NON_RUNNER_FLAGS = {
    "--job-name", "--array", "--gres", "--cpus-per-task", "--mem", "--time",
    "--output", "--error", "--requeue", "--no-requeue", "--partition", "--nodelist",
    "--export",
    "--parsable", "--dependency", "--format", "--query-gpu", "--porcelain",
    "--untracked-files", "--show-toplevel", "--noheader", "--nounits",
    "--ntasks", "--ntasks-per-node", "--nodes", "--exclusive", "--exact", "--constraint",
    "--account", "--qos", "--mail-type", "--mail-user", "--chdir",
}

FLAG = re.compile(r"(?<![\w-])(--[a-z0-9-]+)")
#: SBT- and RBT- both contain "BT-"; only a BT- preceded by neither is stale.
STALE_METHOD = re.compile(r"(?<![SR])\bBT-(SGPR|SVGP)\b")


def declared_flags(path: Path) -> set[str]:
    # Runners quote their options either way.
    return set(re.findall(r"""["'](--[a-z0-9-]+)["']""", path.read_text()))


def job_paths():
    return sorted(SIM_ROOT.rglob("*.sbatch"))


class JobScriptTests(unittest.TestCase):
    def test_every_job_is_declared(self):
        found = {str(p.relative_to(SIM_ROOT)) for p in job_paths()}
        self.assertEqual(sorted(found - set(JOB_RUNNERS)), [],
                         "undeclared job: add it to JOB_RUNNERS")
        self.assertEqual(sorted(set(JOB_RUNNERS) - found), [],
                         "JOB_RUNNERS names a job that no longer exists")

    def test_every_runner_exists(self):
        for job, runners in JOB_RUNNERS.items():
            for runner in runners:
                self.assertTrue((SIM_ROOT / runner).is_file(),
                                f"{job} calls a runner that is gone: {runner}")

    def test_every_job_flag_is_accepted_by_some_runner(self):
        for job, runners in JOB_RUNNERS.items():
            accepted = set()
            for runner in runners:
                accepted |= declared_flags(SIM_ROOT / runner)
            used = set(FLAG.findall((SIM_ROOT / job).read_text()))
            unknown = sorted(used - NON_RUNNER_FLAGS - accepted)
            self.assertEqual(unknown, [], f"{job} passes flags {runners} reject")

    def test_no_job_writes_to_a_retired_runs_tree(self):
        # A launcher writing to runs_v10 while the validator scans runs_v12
        # reports a finished campaign as empty. Every versioned tree named by
        # any job must be one the code still knows about.
        known = set()
        for source, pattern in (
            ("svgp/run_benchmark.py", r"RUNS_DIR\s*=\s*HERE\s*/\s*\"([a-z0-9_]+)\""),
            ("sgpr/run_benchmark.py",
             r"DEFAULT_RUNS_DIR\s*=\s*HERE\s*/\s*\"([a-z0-9_]+)\""),
            ("svgp/analysis/paper_status.py", r'"runs_root":\s*"([a-z0-9_]+)"'),
        ):
            known |= set(re.findall(pattern, (SIM_ROOT / source).read_text()))
        self.assertTrue(known, "no runs root could be read from the code")
        known |= {name.replace("runs_", "logs_") for name in known}
        for path in job_paths() + sorted(SIM_ROOT.rglob("*.sh")):
            used = set(re.findall(r"(?:runs|logs)_(?:gain_)?v\d+", path.read_text()))
            self.assertTrue(
                used <= known,
                f"{path.relative_to(SIM_ROOT)} names {sorted(used - known)}; "
                f"the code knows {sorted(known)}",
            )

    def test_no_job_still_asks_for_the_removed_top_k_predictor(self):
        # Top-k is gone as a predictor; every method now declares sparse,
        # diagonal, block or exact in common/predictors.py.
        for path in job_paths() + sorted(SIM_ROOT.rglob("*.sh")):
            text = path.read_text().lower()
            for token in ("top-k", "top_k", "topk"):
                self.assertNotIn(
                    token, text,
                    f"{path.relative_to(SIM_ROOT)} still references top-k")

    def test_no_job_asks_for_a_method_by_its_retired_name(self):
        for path in job_paths() + sorted(SIM_ROOT.rglob("*.sh")):
            hit = STALE_METHOD.search(path.read_text())
            self.assertIsNone(
                hit,
                f"{path.relative_to(SIM_ROOT)} asks for {hit.group(0) if hit else ''}"
                " — the method is RBT-* now",
            )

    def test_every_launcher_checks_the_tree_the_manifest_records(self):
        # code_state() records source_dirty for the whole of Simulations. A
        # launcher that guards a narrower path set will happily start a run the
        # validator later rejects as dirty -- that is how 280 finished runs
        # were thrown away over 39 leftover figures.
        for path in sorted(SIM_ROOT.rglob("*.sbatch")) + \
                sorted(SIM_ROOT.glob("*.sh")):
            text = path.read_text()
            if "--untracked-files=all" not in text:
                continue
            self.assertNotIn(
                "CODE_PATHS", text,
                f"{path.relative_to(SIM_ROOT)} guards its own path list; the "
                "check must cover all of Simulations, as code_state does")
            self.assertIn(
                'status --porcelain --untracked-files=all -- .', text,
                f"{path.relative_to(SIM_ROOT)} must guard the whole tree")

    def test_every_referenced_sibling_script_exists(self):
        # submit_sbt_refresh_campaign.sh once chained an sbatch that had been
        # deleted; the chain died at the last stage after days of GPU.
        for path in sorted(SIM_ROOT.rglob("*.sh")) + job_paths():
            for name in re.findall(
                    r"\$(?:HERE|SGPR|SVGP)/([A-Za-z0-9_.-]+\.(?:sbatch|sh|py))",
                    path.read_text()):
                self.assertTrue(
                    (path.parent / name).is_file()
                    or (SIM_ROOT / "sgpr" / name).is_file()
                    or (SIM_ROOT / "svgp" / name).is_file(),
                    f"{path.relative_to(SIM_ROOT)} references missing {name}")


if __name__ == "__main__":
    unittest.main()


class SubmissionHygieneTests(unittest.TestCase):
    """Nothing in the artifact may name a tool, a person, or one machine."""

    #: Files a reviewer or artifact evaluator reads. Binary and cache paths are
    #: excluded; everything else in the tree is source as far as this is
    #: concerned.
    SUFFIXES = (".py", ".sh", ".sbatch", ".md", ".toml", ".txt", ".cfg", ".tex")

    FORBIDDEN = (
        # Assistant and tool names have no place in a submitted artifact.
        "codex", "claude", "anthropic", "openai", "chatgpt", "copilot",
        # Nor does one author's machine: an evaluator cannot run these paths.
        "/users/", "/export/home", "macbook",
    )

    def _sources(self):
        for path in sorted(SIM_ROOT.rglob("*")):
            if not path.is_file() or path.suffix not in self.SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            # This file necessarily spells the words it bans.
            if path.resolve() == Path(__file__).resolve():
                continue
            yield path

    def test_no_tool_or_machine_names_anywhere(self):
        hits = []
        for path in self._sources():
            text = path.read_text(errors="replace").lower()
            for word in self.FORBIDDEN:
                if word in text:
                    hits.append(f"{path.relative_to(SIM_ROOT)}: {word}")
        self.assertEqual(hits, [], "submission artifact names a tool or a machine")

    def test_every_launcher_resolves_its_environment_relatively(self):
        # A launcher that hardcodes one user's virtualenv runs on exactly one
        # machine. Each must derive the path from its own location.
        for path in sorted(SIM_ROOT.rglob("*.sh")) + sorted(SIM_ROOT.rglob("*.sbatch")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not any(k in line for k in ("VENV", "ACTIVATE", "venv")):
                    continue
                self.assertNotRegex(
                    line, r'\$HOME/[A-Za-z]',
                    f"{path.relative_to(SIM_ROOT)}:{number} resolves an "
                    "environment under a user's $HOME; anchor it to the "
                    "script's own location")
