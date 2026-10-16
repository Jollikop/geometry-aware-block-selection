#!/usr/bin/env python3
"""Freeze a completed campaign, and check whether later commits can affect it.

Freezing means three things at once:

1. the campaign is release-ready under ``paper_status`` for its own plan;
2. its results are copied to a read-only directory with a manifest recording
   the commit, the per-run fingerprints, the data signatures and the package
   profile;
3. the set of source files the campaign actually depends on is declared, so
   that a later commit can be checked against it instead of being trusted.

Point 3 is what allows the non-stationary model to keep changing after the
stationary campaign is frozen: ``common/kernels_nonstationary.py`` is not in
the stationary dependency set, so edits confined to it provably cannot change
a frozen stationary result. ``--verify`` turns that from a claim into a
command that prints the offending files if there are any.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
REPO_ROOT = SIM_ROOT.parent

from svgp.analysis import paper_status as ps  # noqa: E402

# Files whose content can change a result of the given plan. Paths are
# relative to Simulations/. The non-stationary module appears only where the
# kernel is actually reachable.
STATIONARY_DEPENDENCIES = (
    "svgp/run_benchmark.py",
    "svgp/engine.py",
    "svgp/plans.py",
    "common/datasets.py",
    "common/data_split.py",
    "common/kernels.py",
    # Every SBT partition comes from here, so a change to it changes every SBT
    # result. Omitting it once made --verify report UNAFFECTED for a commit
    # that moved every number in a frozen campaign.
    "common/partitions.py",
    "common/terms.py",
    "common/predictors.py",
    "common/provenance.py",
)
PLAN_DEPENDENCIES = {
    "report": STATIONARY_DEPENDENCIES,
    "grid": STATIONARY_DEPENDENCIES,
    "grid-exploratory": STATIONARY_DEPENDENCIES
    + ("common/kernels_nonstationary.py",),
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_state(plan: str, run_tag: str, runs_dir: Path | None):
    root = runs_dir or HERE / ps.PLAN_PROFILE[plan]["runs_root"] / run_tag
    pairs = ps.campaign_pairs(plan)
    expected = ps.expected_keys(pairs, plan)
    complete, invalid, duplicates, commits = ps.scan_results(
        root, run_tag, expected, plan
    )
    ready = (
        len(complete) == len(expected)
        and invalid == 0 and duplicates == 0 and len(commits) == 1
    )
    return root, expected, complete, invalid, duplicates, commits, ready


def verify(plan: str, frozen_commit: str, against: str) -> int:
    """Report source changes that could invalidate the frozen campaign."""
    deps = PLAN_DEPENDENCIES[plan]
    changed = git(
        "diff", "--name-only", frozen_commit, against, "--",
        *[str((SIM_ROOT / dep).relative_to(REPO_ROOT)) for dep in deps],
    )
    changed = [line for line in changed.split("\n") if line.strip()]
    print(f"plan={plan} frozen={frozen_commit[:12]} against={against[:12]}")
    print(f"declared dependencies: {len(deps)}")
    if not changed:
        print("UNAFFECTED: no declared dependency changed between the two "
              "commits; the frozen results remain valid.")
        return 0
    print("AFFECTED: the following declared dependencies changed:")
    for path in changed:
        print(f"  {path}")
    print("The frozen campaign must be re-run, or the change reverted.")
    return 1


def freeze(plan: str, run_tag: str, runs_dir: Path | None,
           out: Path, force: bool) -> int:
    root, expected, complete, invalid, duplicates, commits, ready = (
        campaign_state(plan, run_tag, runs_dir)
    )
    print(f"complete={len(complete)}/{len(expected)} invalid={invalid} "
          f"duplicates={duplicates} commits={len(commits)}")
    if not ready and not force:
        print("NOT release-ready: refusing to freeze (use --force to "
              "override for a partial snapshot).")
        return 2
    if out.exists():
        print(f"refusing to overwrite existing {out}")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, out / "runs")

    frozen_commit = next(iter(commits)) if len(commits) == 1 else None
    manifest = {
        "plan": plan,
        "run_tag": run_tag,
        "source_runs_dir": str(root),
        "code_commit": frozen_commit,
        "head_at_freeze": git("rev-parse", "HEAD"),
        "declared_dependencies": list(PLAN_DEPENDENCIES[plan]),
        "dependency_sha256": {
            dep: sha256(SIM_ROOT / dep) for dep in PLAN_DEPENDENCIES[plan]
        },
        "expected_results": len(expected),
        "complete_results": len(complete),
        "invalid": invalid,
        "duplicates": duplicates,
        "release_ready": bool(ready),
    }
    fingerprints = {}
    for metrics_path in sorted((out / "runs").rglob("metrics.json")):
        result = json.loads(metrics_path.read_text())
        fingerprints[str(metrics_path.relative_to(out))] = {
            "config_id": result.get("config_id"),
            "data_signature": (result.get("config") or {}).get(
                "data_signature"
            ),
        }
    manifest["runs"] = fingerprints
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(f"frozen {len(fingerprints)} runs -> {out}")
    print(f"campaign commit: {frozen_commit}")
    print("Tag this state, e.g.:")
    print(f"  git tag frozen-{run_tag} {frozen_commit}")
    return 0 if ready else 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", choices=tuple(PLAN_DEPENDENCIES),
                    default="grid")
    ap.add_argument("--run-tag", default="paper_grid_v10")
    ap.add_argument("--runs-dir", type=Path)
    ap.add_argument("--out", type=Path,
                    help="destination of the frozen copy")
    ap.add_argument("--force", action="store_true",
                    help="freeze even if the campaign is not release-ready")
    ap.add_argument("--verify", action="store_true",
                    help="do not freeze; check a commit against the declared "
                         "dependency set")
    ap.add_argument("--frozen-commit",
                    help="commit of the frozen campaign (default: read from "
                         "the manifest under --out)")
    ap.add_argument("--against", default="HEAD",
                    help="commit to check against (default: HEAD)")
    args = ap.parse_args()

    if args.verify:
        frozen = args.frozen_commit
        if not frozen and args.out and (args.out / "manifest.json").exists():
            frozen = json.loads(
                (args.out / "manifest.json").read_text()
            )["code_commit"]
        if not frozen:
            raise SystemExit(
                "--verify needs --frozen-commit or an --out with a manifest"
            )
        raise SystemExit(verify(args.plan, frozen, args.against))

    out = args.out or HERE / "frozen" / args.run_tag
    raise SystemExit(
        freeze(args.plan, args.run_tag, args.runs_dir, out, args.force)
    )


if __name__ == "__main__":
    main()
