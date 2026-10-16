#!/usr/bin/env python3
"""Validate and optionally freeze the complete paper experiment suite.

This validator deliberately owns the cross-campaign contract. Individual
launchers may resume a partial tree, but a final suite is accepted only when
every expected identity occurs once, all reported metrics are finite, every
run comes from one clean commit, and the artefacts needed for post-hoc
prediction are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# Imported, not restated. A validator that keeps its own copy of the contract
# validates its copy, not the code -- and the copies drift. Both modules are
# numpy-or-lighter, so this stays runnable on a host without JAX.
from common.predictors import (
    COLLAPSED_METHOD_PREDICTOR,
    VARIATIONAL_METHOD_PREDICTOR,
)
from common.provenance import config_fingerprint, sha256_file
from sgpr import protocol
from svgp import plans


#: Table order: RBF first, Gibbs second, the remaining two after them.
KERNELS = ("rbf", "gibbs", "matern52", "amplitude-rbf")
DATASETS = tuple(plans.GRID_DATASETS)
M_VALUES = tuple(plans.REPORT_DEFAULT_M_VALUES)
SEEDS = tuple(plans.GRID_SEEDS)
SGPR_METHODS = (
    "Exact-GP", "E-SGPR", "METIS-SGPR", "SGPR", "T-SGPR",
    "RBT-SGPR", "SBT-SGPR",
)
SVGP_METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
#: The registries carry retired aliases for reading archived runs; a final
#: suite only ever contains the current names.
SGPR_PREDICTOR = {
    method: COLLAPSED_METHOD_PREDICTOR[method] for method in SGPR_METHODS
}
SVGP_PREDICTOR = {
    method: VARIATIONAL_METHOD_PREDICTOR[method] for method in SVGP_METHODS
}


@dataclass
class Check:
    name: str
    expected: int
    complete: int = 0
    invalid: int = 0


#: The fingerprint the runners write is common.provenance.config_fingerprint;
#: a second implementation here can only ever disagree with it.
canonical_fingerprint = config_fingerprint


def finite(row: dict, names: tuple[str, ...]) -> bool:
    try:
        return all(math.isfinite(float(row[name])) for name in names)
    except (KeyError, TypeError, ValueError):
        return False


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def exact_paths(root: Path, expected: set[tuple[str, str, int, str, int]]):
    for kernel, dataset, m, method, seed in sorted(expected):
        yield (
            (kernel, dataset, m, method, seed),
            root / kernel / dataset / f"M{m}" / method / f"seed{seed}",
        )


def check_svgp_tree(
    name: str,
    root: Path,
    kernels: tuple[str, ...],
    *,
    epochs: int | None = None,
    transplant: bool = False,
    history: bool = False,
) -> tuple[Check, set[str]]:
    methods = tuple(
        f"{method}-to-SVGP" for method in SVGP_METHODS
    ) if transplant else SVGP_METHODS
    expected = {
        (kernel, dataset, m, method, seed)
        for kernel in kernels for dataset in DATASETS for m in M_VALUES
        for method in methods for seed in SEEDS
    }
    check = Check(name, len(expected))
    commits: set[str] = set()
    expected_files: set[Path] = set()
    for identity, run_dir in exact_paths(root, expected):
        metrics_path = run_dir / "metrics.json"
        expected_files.add(metrics_path)
        row = read_json(metrics_path)
        if row is None:
            continue
        kernel, dataset, m, method, seed = identity
        config = row.get("config") or {}
        donor = method.removesuffix("-to-SVGP") if transplant else method
        predictor = "sparse" if transplant else SVGP_PREDICTOR[method]
        valid = (
            (row.get("kernel"), row.get("dataset"), row.get("M"),
             row.get("method"), row.get("seed")) == identity
            and finite(row, (
                "rmse", "nll",
                "last_epoch_stochastic_objective_estimate_per_datum",
            ))
            and row.get("reported_predictor") == predictor
            and config.get("reported_predictor") == predictor
            and config.get("code_source_dirty") is False
            and row.get("config_id") == canonical_fingerprint(config)
            and (run_dir / "params.npz").is_file()
        )
        valid = valid and (
            config.get("optimizer") == "adam"
            and config.get("q_optimizer") == "natural"
            and float(config.get("natural_lr", -1.0)) == 0.05
            and float(config.get("weight_decay", -1.0)) == 0.0
            and config.get("lr_schedule") == "constant"
            and config.get("penalty_sampling") == "uniform"
            and config.get("batch_coupling") == "split"
        )
        if epochs is not None:
            valid = valid and int(config.get("epochs", -1)) == epochs
        if not transplant and method in {"RBT-SVGP", "SBT-SVGP"}:
            valid = valid and (run_dir / "final_blocks.npz").is_file()
        if transplant:
            valid = valid and (
                row.get("training_method") == "SVGP"
                and row.get("donor_method") == donor
                and config.get("training_method") == "SVGP"
                and config.get("donor_method") == donor
                and config.get("transplant_to_svgp") is True
                and config.get("phases") == "10:theta,z 140:theta"
                and config.get("metrics_at_epochs") == [0, 10, 25, 50, 100, 150]
                and config.get("lr") == 0.001
                and config.get("z_lr") == 0.001
                and config.get("natural_lr") == 0.05
                and str(config.get("donor_params_sha256", "")).startswith(
                    "sha256:"
                )
            )
        else:
            valid = valid and (
                float(config.get("lr", -1.0)) == 0.005
                and float(config.get("z_lr", -1.0)) == 0.005
            )
        if history:
            valid = valid and (run_dir / "history.csv").is_file()
        if valid:
            check.complete += 1
            commits.add(str(config.get("code_commit", "")))
        else:
            check.invalid += 1

    extras = set(root.rglob("metrics.json")) - expected_files if root.exists() else set()
    check.invalid += len(extras)
    return check, commits


def check_convergence(root: Path) -> tuple[Check, set[str]]:
    expected = {
        (kernel, "kin40k", 256, method, 0)
        for kernel in KERNELS for method in SVGP_METHODS
    }
    check = Check("svgp-convergence", len(expected))
    commits: set[str] = set()
    expected_files: set[Path] = set()
    for identity, run_dir in exact_paths(root, expected):
        path = run_dir / "metrics.json"
        expected_files.add(path)
        row = read_json(path)
        if row is None:
            continue
        config = row.get("config") or {}
        kernel, _, _, method, _ = identity
        expected_epochs = 350 if kernel in {"rbf", "matern52"} else 500
        valid = (
            (row.get("kernel"), row.get("dataset"), row.get("M"),
             row.get("method"), row.get("seed")) == identity
            and finite(row, ("rmse", "nll"))
            and int(config.get("epochs", -1)) == expected_epochs
            and config.get("record_history") is True
            and (run_dir / "history.csv").is_file()
            and (run_dir / "params.npz").is_file()
            and config.get("code_source_dirty") is False
            and row.get("config_id") == canonical_fingerprint(config)
        )
        valid = valid and (
            config.get("optimizer") == "adam"
            and float(config.get("lr", -1.0)) == 0.005
            and float(config.get("z_lr", -1.0)) == 0.005
            and config.get("q_optimizer") == "natural"
            and float(config.get("natural_lr", -1.0)) == 0.05
            and float(config.get("weight_decay", -1.0)) == 0.0
            and config.get("lr_schedule") == "constant"
            and config.get("penalty_sampling") == "uniform"
            and config.get("batch_coupling") == "split"
        )
        if valid:
            check.complete += 1
            commits.add(str(config.get("code_commit", "")))
        else:
            check.invalid += 1
    extras = set(root.rglob("metrics.json")) - expected_files if root.exists() else set()
    check.invalid += len(extras)
    return check, commits


def check_sgpr(
    root: Path, *, m_values: tuple[int, ...] = (256,),
) -> tuple[Check, set[str]]:
    expected = {
        (kernel, "kin40k", m, method, seed)
        for kernel in KERNELS for m in m_values
        for method in SGPR_METHODS for seed in SEEDS
    }
    check = Check("sgpr-main", len(expected))
    commits: set[str] = set()
    expected_files: set[Path] = set()
    for identity, run_dir in exact_paths(root, expected):
        path = run_dir / "metrics.json"
        expected_files.add(path)
        row = read_json(path)
        if row is None:
            continue
        config = row.get("config") or {}
        method = identity[3]
        valid = (
            (row.get("kernel"), row.get("dataset"), row.get("M"),
             row.get("method"), row.get("seed")) == identity
            and finite(row, ("rmse", "nll", "obj_n", "elbo_n"))
            and row.get("reported_predictor") == SGPR_PREDICTOR[method]
            and config.get("reported_predictor") == SGPR_PREDICTOR[method]
            and config.get("optimizer") == "lbfgs"
            and config.get("optimizer_backend") == "scipy-l-bfgs-b"
            and float(config.get("scipy_lbfgsb_ftol", -1.0))
                == protocol.SCIPY_LBFGSB_FTOL
            and float(config.get("scipy_lbfgsb_gtol", -1.0))
                == protocol.SCIPY_LBFGSB_GTOL
            and int(config.get("scipy_lbfgsb_maxls", -1))
                == protocol.SCIPY_LBFGSB_MAXLS
            and int(config.get("scipy_lbfgsb_maxcor", -1))
                == protocol.SCIPY_LBFGSB_MAXCOR
            and int(config.get("scipy_lbfgsb_maxfun", -1))
                == protocol.SCIPY_LBFGSB_MAXFUN
            and int(config.get("n_train", -1)) == 5000
            and int(config.get("n_test", -1)) == 2000
            and int(config.get("steps", -1)) == protocol.LBFGS_STEPS
            and config.get("refresh_schedule") == list(
                protocol.LBFGS_REFRESH_SCHEDULE
            )
            and config.get("require_convergence") is True
            and int(config.get("convergence_window_steps", -1))
                == protocol.CONVERGENCE_WINDOW_STEPS
            and float(config.get("convergence_obj_n_tol", -1.0))
                == protocol.CONVERGENCE_OBJ_N_TOL
            and float(config.get("convergence_grad_max_n_tol", -1.0))
                == protocol.CONVERGENCE_GRAD_MAX_N_TOL
            and config.get("initialization_policy")
                == list(protocol.INITIALIZATION_POLICY)
            and int(config.get("initialization_nested_max_m", -1))
                == protocol.INITIALIZATION_NESTED_MAX_M
            and int(config.get("warmup_max_cycles", -1))
                == protocol.WARMUP_MAX_CYCLES
            and int(config.get("warmup_theta_steps", -1))
                == protocol.WARMUP_THETA_STEPS
            and float(config.get("warmup_reselect_obj_n_tol", -1.0))
                == protocol.WARMUP_RESELECT_OBJ_N_TOL
            and int(config.get("convergence_patience", -1))
                == protocol.CONVERGENCE_PATIENCE
            and row.get("convergence_passed") is True
            and row.get("optimizer_convergence_reached") is True
            and 1 <= int(row.get("steps", -1)) <= protocol.LBFGS_STEPS
            and int(row.get("max_steps", -1)) == protocol.LBFGS_STEPS
            and row.get("optimizer_backend") == "scipy-l-bfgs-b"
            and row.get("initialization_strategy") in (
                *protocol.INITIALIZATION_POLICY, "not-applicable"
            )
            and isinstance(row.get("optimization_attempts"), list)
            and 0 <= int(row.get("restart_count", -1)) \
                < len(protocol.INITIALIZATION_POLICY)
            and 0 <= int(row.get("warmup_cycles", -1)) \
                <= protocol.WARMUP_MAX_CYCLES
            and (
                (
                    len(row.get("convergence_checks") or [])
                        >= protocol.CONVERGENCE_PATIENCE
                    and all(
                        check_item.get("stable") is True
                        and int(check_item.get("budget", -1))
                            == protocol.CONVERGENCE_WINDOW_STEPS
                        for check_item in (row.get(
                            "convergence_checks"
                        ) or [])[-protocol.CONVERGENCE_PATIENCE:]
                    )
                )
                or (
                    row.get("optimizer_terminal_convergence") is True
                    and bool(row.get("convergence_checks"))
                    and row["convergence_checks"][-1].get(
                        "solver_terminal"
                    ) is True
                    and 0 <= int(row["convergence_checks"][-1].get(
                        "budget", -1
                    )) <= protocol.CONVERGENCE_WINDOW_STEPS
                    and row["convergence_checks"][-1].get("stable") is True
                )
            )
            and float(row.get("convergence_tail_range_obj_n", math.inf))
                <= protocol.CONVERGENCE_OBJ_N_TOL
            and finite(row, (
                "convergence_tail_delta_obj_n",
                "convergence_tail_range_obj_n",
                "final_grad_l2_per_datum",
                "final_grad_max_per_datum",
            ))
            and float(row.get("final_grad_max_per_datum", math.inf))
                <= protocol.CONVERGENCE_GRAD_MAX_N_TOL
            and int(config.get("block_size", -1)) == 500
            and int(config.get("M", -1)) == identity[2]
            and config.get("code_source_dirty") is False
            and row.get("config_id") == canonical_fingerprint(config)
            and (run_dir / "params.npz").is_file()
        )
        if method in {"METIS-SGPR", "RBT-SGPR", "SBT-SGPR"}:
            valid = valid and (run_dir / "final_blocks.npz").is_file()
        if valid:
            check.complete += 1
            commits.add(str(config.get("code_commit", "")))
        else:
            check.invalid += 1
    extras = set(root.rglob("metrics.json")) - expected_files if root.exists() else set()
    check.invalid += len(extras)
    return check, commits


def check_sgpr_transplant(
    root: Path,
    *,
    m_values: tuple[int, ...] = (256,),
    nested_m: bool = False,
) -> tuple[Check, set[str]]:
    check = Check("sgpr-transplant", 40 * len(m_values))
    commits: set[str] = set()
    for m in m_values:
        for kernel in KERNELS:
            kernel_root = (root / f"M{m}" if nested_m else root) / kernel
            for seed in SEEDS:
                path = kernel_root / f"seed{seed}" / "metrics.json"
                payload = read_json(path)
                if payload is None:
                    continue
                rows = payload.get("rows") or []
                code = payload.get("code") or {}
                methods = {row.get("donor_method") for row in rows}
                sgpr_control = next(
                    (row for row in rows if row.get("donor_method") == "SGPR"),
                    None,
                )
                valid = (
                    payload.get("implementation")
                        == "sgpr-hyperparameter-transplant-v7"
                    and len(rows) == 4
                    and methods
                        == {"SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR"}
                    and all(finite(row, (
                        "donor_complete_nll", "donor_sparse_nll",
                        "transplant_sgpr_nll", "transplant_sgpr_elbo_n",
                    )) for row in rows)
                    and all(row.get("kernel") == kernel for row in rows)
                    and all(int(row.get("seed", -1)) == seed for row in rows)
                    and all(int(row.get("M", -1)) == m for row in rows)
                    and int(payload.get("config", {}).get("M", -1)) == m
                    and payload.get("config", {}).get("donor_runs_dir")
                    and int(payload.get("config", {}).get(
                        "transplant_steps", -1
                    )) == protocol.TRANSPLANT_STEPS
                    and int(payload.get("config", {}).get(
                        "convergence_window_steps", -1
                    )) == protocol.CONVERGENCE_WINDOW_STEPS
                    and float(payload.get("config", {}).get(
                        "convergence_obj_n_tol", -1.0
                    )) == protocol.CONVERGENCE_OBJ_N_TOL
                    and float(payload.get("config", {}).get(
                        "convergence_grad_max_n_tol", -1.0
                    )) == protocol.CONVERGENCE_GRAD_MAX_N_TOL
                    and int(payload.get("config", {}).get(
                        "convergence_patience", -1
                    )) == protocol.CONVERGENCE_PATIENCE
                    and payload.get("config", {}).get("optimizer_backend")
                        == "scipy-l-bfgs-b"
                    and float(payload.get("config", {}).get(
                        "scipy_lbfgsb_ftol", -1.0
                    )) == protocol.SCIPY_LBFGSB_FTOL
                    and float(payload.get("config", {}).get(
                        "scipy_lbfgsb_gtol", -1.0
                    )) == protocol.SCIPY_LBFGSB_GTOL
                    and int(payload.get("config", {}).get(
                        "scipy_lbfgsb_maxls", -1
                    )) == protocol.SCIPY_LBFGSB_MAXLS
                    and int(payload.get("config", {}).get(
                        "scipy_lbfgsb_maxcor", -1
                    )) == protocol.SCIPY_LBFGSB_MAXCOR
                    and int(payload.get("config", {}).get(
                        "scipy_lbfgsb_maxfun", -1
                    )) == protocol.SCIPY_LBFGSB_MAXFUN
                    and all(
                        row.get("transplant_convergence_reached") is True
                        and 1 <= int(row.get("transplant_steps", -1))
                            <= protocol.TRANSPLANT_STEPS
                        and int(row.get("transplant_max_steps", -1))
                            == protocol.TRANSPLANT_STEPS
                        and float(row.get(
                            "transplant_final_grad_max_per_datum", math.inf
                        )) <= protocol.CONVERGENCE_GRAD_MAX_N_TOL
                        for row in rows
                    )
                    and sgpr_control is not None
                    and abs(float(sgpr_control.get(
                        "delta_sgpr_elbo_n", math.inf
                    ))) <= protocol.TRANSPLANT_NULL_ELBO_TOL
                    and abs(float(sgpr_control.get(
                        "delta_sparse_nll", math.inf
                    ))) <= protocol.TRANSPLANT_NULL_NLL_TOL
                    and code.get("source_dirty") is False
                    and all(
                        (path.parent / f"params_transplant_{method}.npz").is_file()
                        for method in (
                            "SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR"
                        )
                    )
                )
                if valid:
                    check.complete += 1
                    commits.add(str(code.get("commit", "")))
                else:
                    check.invalid += 1
    return check, commits


def check_auxiliary(root: Path) -> tuple[Check, set[str]]:
    check = Check("auxiliary-figures", 2)
    residual = root / "sgpr" / "residual_illustration"
    residual_files = (
        residual / "residual_structure_kin40k_rbf_gibbs.pdf",
        residual / "residual_structure_kin40k_rbf_gibbs.png",
        residual / "residual_structure_kin40k_rbf_gibbs_results.csv",
        residual / "kin40k_rbf_residual.npz",
        residual / "kin40k_gibbs_residual.npz",
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in residual_files):
        check.complete += 1
    snelson = root / "sgpr" / "snelson"
    snelson_files = (
        snelson / "snelson_sgpr_t_rbt_sbt.pdf",
        snelson / "snelson_summary_sgpr_t_rbt_sbt.csv",
        snelson / "snelson_table_sgpr_t_rbt_sbt.tex",
        snelson / "snelson_sgpr_t_rbt_sbt_manifest.json",
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in snelson_files):
        manifest = read_json(snelson_files[-1])
        if manifest and (manifest.get("code") or {}).get("source_dirty") is False:
            check.complete += 1
    return check, set()


def sha256(path: Path) -> str:
    """Bare hex digest.

    ``common.provenance.sha256_file`` returns a prefixed ``sha256:...``
    signature for manifests; the freeze index compares bare hex, so the two are
    deliberately different shapes rather than a duplicate.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(root: Path, checks: list[Check], commit: str, scope: str) -> None:
    manifest_path = root / "FROZEN_MANIFEST.json"
    sums_path = root / "SHA256SUMS.txt"
    if manifest_path.exists() or sums_path.exists():
        raise SystemExit("suite already has frozen metadata; refusing overwrite")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path not in {manifest_path, sums_path}
    )
    sums_path.write_text("".join(
        f"{sha256(path)}  {path.relative_to(root)}\n" for path in files
    ))
    manifest = {
        "schema": f"aistats2026-{scope}-suite-v1",
        "code_commit": commit,
        "checks": {
            check.name: {
                "expected": check.expected,
                "complete": check.complete,
                "invalid": check.invalid,
            }
            for check in checks
        },
        "sha256_file_count": len(files),
        "read_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(f"frozen {len(files)} files at commit {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--scope", choices=("full", "sgpr", "sgpr-sweep"), default="full"
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.scope == "sgpr-sweep":
        sgpr_results: list[tuple[Check, set[str]]] = [
            check_sgpr(root / "runs", m_values=(128, 256, 512)),
            check_sgpr_transplant(
                root / "transplant",
                m_values=(128, 256, 512),
                nested_m=True,
            ),
        ]
    else:
        sgpr_results = [
            check_sgpr(root / "sgpr" / "runs"),
            check_sgpr_transplant(root / "sgpr" / "transplant"),
            check_auxiliary(root),
        ]
    full_only: list[tuple[Check, set[str]]] = [
        check_svgp_tree(
            "svgp-stationary", root / "svgp" / "stationary" / "runs",
            ("rbf", "matern52"), epochs=350,
        ),
        check_svgp_tree(
            "svgp-extended", root / "svgp" / "extended" / "runs",
            ("amplitude-rbf", "gibbs"), epochs=500,
        ),
        check_svgp_tree(
            "svgp-transplant", root / "svgp" / "transplant" / "runs",
            KERNELS, epochs=150, transplant=True, history=True,
        ),
        check_convergence(root / "svgp" / "convergence" / "runs"),
    ]
    results = (
        sgpr_results
        if args.scope in {"sgpr", "sgpr-sweep"}
        else sgpr_results + full_only
    )
    checks = [item[0] for item in results]
    commits = set().union(*(item[1] for item in results))
    ready = (
        all(check.complete == check.expected and check.invalid == 0
            for check in checks)
        and len(commits) == 1
        and next(iter(commits), "") not in {"", "None"}
    )
    if args.brief:
        complete = sum(check.complete for check in checks)
        expected = sum(check.expected for check in checks)
        invalid = sum(check.invalid for check in checks)
        print(
            f"complete={complete}/{expected} invalid={invalid} "
            f"commits={len(commits)} release_ready={'yes' if ready else 'no'}"
        )
    else:
        for check in checks:
            print(
                f"{check.name:20s} {check.complete:5d}/{check.expected:<5d} "
                f"invalid={check.invalid}"
            )
        print(f"commits={sorted(commits)}")
        print(f"release_ready={'yes' if ready else 'no'}")
    if args.freeze:
        if not ready:
            raise SystemExit("suite is not release-ready; refusing to freeze")
        freeze(root, checks, next(iter(commits)), args.scope)
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
