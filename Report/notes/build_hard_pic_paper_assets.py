"""Certify downloaded hard-block predictions and build paper-only assets.

No training record, frozen prediction, or historical figure is modified.
The sparse-transplant curves and stochastic objectives retain their certified
sources; only complete-predictor RMSE/NLL cells are reaggregated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / "Report"
DEFAULT_INPUT = PROJECT / "output/results/hard_pic_predict_full_v1_20261003"
DEFAULT_OUTPUT = REPORT / "Results/hard_pic_full_v1"
DEFAULT_FIGURES = REPORT / "figures/hard_pic_full_v1"
METHODS = ("T-SVGP", "RBT-SVGP", "SBT-SVGP")
KERNELS = ("rbf", "matern52", "amplitude-rbf", "gibbs")
DATASETS = ("pol", "superconductivity", "kin40k", "protein",
            "elevators", "traffic", "electricity", "3droad")
MS = (64, 128, 256, 512)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPORT / "notes" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mean_sem(values: list[float]) -> tuple[float, float]:
    assert len(values) == 10 and all(math.isfinite(v) for v in values)
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(10)


def load_predictions(root: Path):
    plan_path = root / "plan.json"
    certificate_path = root / "HARD_PIC_PREDICTION_CERTIFICATE.json"
    plan = json.loads(plan_path.read_text())
    certificate = json.loads(certificate_path.read_text())
    assert plan["implementation"] == "hard-pic-pitc-plan-v1"
    assert certificate["implementation"] == "hard-block-pic-prediction-v1"
    assert certificate["plan_sha256"] == digest(plan_path)
    assert certificate["count"] == len(plan["runs"]) == 3880
    assert len(certificate["predictions"]) == 3880
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    minimum_raw = float("inf")
    for item in plan["runs"]:
        relative = item["output_relative"]
        path = root / "hard" / relative / "prediction.json"
        assert digest(path) == certificate["predictions"][relative], path
        row = json.loads(path.read_text())
        identity = row["identity"]
        assert row["valid"] is True, path
        assert row["interpretation"] == certificate["interpretation"]
        assert identity["implementation"] == certificate["implementation"]
        assert identity["plan_sha256"] == certificate["plan_sha256"]
        assert identity["evaluator_sha256"] == plan["hard_evaluator_sha256"]
        assert identity["source_commit"] == plan["expected_heads"][
            "sgpr" if item["family"] == "sgpr" else "svgp"]
        key = (item["family"], item["kernel"], item["dataset"],
               int(item["M"]), item["method"], int(item["seed"]))
        assert key not in seen, key
        seen.add(key)
        assert all(row[field] == item[field] for field in
                   ("family", "kernel", "dataset", "method", "M", "seed"))
        assert row["diagnostics"]["negative_raw_residual_count"] == 0
        minimum_raw = min(minimum_raw, row["diagnostics"]["min_raw_residual"])
        for metric in ("rmse", "nll"):
            value = float(row[metric])
            assert math.isfinite(value), path
            groups[key[:-1]][metric].append(value)
    expected_svgp = set(itertools.product(
        ("svgp",), KERNELS, DATASETS, MS, METHODS, range(10)))
    expected_sgpr = set(itertools.product(
        ("sgpr",), ("rbf",), ("kin40k",), (512,),
        ("T-SGPR", "RBT-SGPR", "SBT-SGPR", "METIS-SGPR"), range(10)))
    assert seen == expected_svgp | expected_sgpr
    assert len(groups) == 388 and minimum_raw >= -1e-7
    return groups, dict(plan_sha256=digest(plan_path),
                        certificate_sha256=digest(certificate_path),
                        prediction_files=len(seen), minimum_raw_residual=minimum_raw)


def sgpr_table(groups, output: Path):
    with (REPORT / "Results/kin40k_sgpr_main_summary.csv").open(newline="") as stream:
        frozen = {row["method"]: row for row in csv.DictReader(stream)
                  if row["kernel"] == "rbf" and int(row["M"]) == 512}
    order = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR",
             "METIS-SGPR", "Exact-GP")
    label = {"RBT-SGPR": "(R)BT-SGPR", "SBT-SGPR": "(S)BT-SGPR",
             "METIS-SGPR": "(METIS)BT-SGPR$^\\dagger$",
             "Exact-GP": "Exact GP$^\\dagger$"}
    values = {}
    for method in order:
        baseline = frozen[method]
        assert int(baseline["n_seeds"]) == 10
        rmse = (float(baseline["rmse_mean"]),
                float(baseline["rmse_std"]) / math.sqrt(10))
        if method in ("SGPR", "Exact-GP"):
            nll = (float(baseline["nll_mean"]),
                   float(baseline["nll_std"]) / math.sqrt(10))
        else:
            sample = groups[("sgpr", "rbf", "kin40k", 512, method)]
            assert math.isclose(mean_sem(sample["rmse"])[0], rmse[0], abs_tol=1e-8)
            nll = mean_sem(sample["nll"])
        values[method] = rmse, nll
    def cell(pair, best=False):
        value, sem = pair
        result = f"${value:.3f}({round(1000 * sem)})$"
        return "\\textbf{" + result + "}" if best else result
    best_rmse = min(values[m][0][0] for m in order[:4])
    best_nll = min(values[m][1][0] for m in order[:4])
    lines = [
        "% Prediction-only hard-block certificate; frozen fits and search times unchanged.",
        r"\begin{table}[!ht]",
        r"\caption{RBF--Kin40k: \(M=512\), \(N=5000\), 2000 test points and "
        r"\(b=500\). Test RMSE/NLL use the block-diagonal predictor "
        r"of Eq.~\eqref{eq:hard-block-correction}; mean (SE in parentheses) "
        r"over ten seeds. Search is the original partition-build time in ms. "
        r"Bold: best non-dense method; \(^\dagger\)dense reference.}",
        r"\label{tab:kin40k-sgpr}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Method & RMSE & NLL & Search (ms) \\", r"\midrule",
    ]
    for method in order:
        rmse, nll = values[method]
        search = ("6.07" if method == "SBT-SGPR" else
                  "1034.92" if method == "METIS-SGPR" else "---")
        lines.append(" & ".join((label.get(method, method),
            cell(rmse, method in order[:4] and math.isclose(rmse[0], best_rmse)),
            cell(nll, method in order[:4] and math.isclose(nll[0], best_nll)),
            search)) + r" \\")
        if method == "SBT-SGPR":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    output.write_text("\n".join(lines))
    return {method: {"rmse": values[method][0], "nll": values[method][1]}
            for method in order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    groups, prediction_provenance = load_predictions(args.input)
    appendix = load_module("frozen_appendix", "build_svgp_frozen_appendix.py")
    focused = load_module("focused_page", "build_focused_svgp_page.py")
    table, frozen_provenance = appendix.load()
    for (family, kernel, dataset, m, method), metrics in groups.items():
        if family != "svgp":
            continue
        for metric in ("rmse", "nll"):
            value = mean_sem(metrics[metric])
            key = ("complete", kernel, dataset, m, method, metric)
            if metric == "rmse":
                assert math.isclose(value[0], table[key][0], abs_tol=1e-8), key
                assert math.isclose(value[1], table[key][1], abs_tol=1e-8), key
            table[key] = value
    assert len(table) == 2560
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    summary = args.output / "svgp_hard_pic_summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "kernel", "dataset", "M", "method",
                         "metric", "n", "mean", "sem"))
        for key, value in sorted(table.items()):
            writer.writerow((*key, 10, *value))
    figure_paths = []
    for kernel in KERNELS:
        for group in range(2):
            path = args.figures / f"{kernel}_{group + 1}.pdf"
            appendix.draw(table, kernel, DATASETS[group * 4:(group + 1) * 4], path)
            figure_paths.append(path)
    main_figure = args.figures / "svgp_rbf_amplitude_final_page.pdf"
    focused.campaign(table_override=table, output=main_figure, wide=False)
    figure_paths.append(main_figure)
    snelson = args.input / "snelson"
    snelson_record = json.loads((snelson / "snelson_hard_prediction.json").read_text())
    assert snelson_record["fit_reproduced_within_1e-4_per_datum"] is True
    assert digest(snelson / "snelson_hard_prediction.pdf") == (
        snelson_record["hard_figure_sha256"])
    snelson_figure = REPORT / "figures/snelson_prediction_hard_pic_readable.pdf"
    assert snelson_figure.is_file()
    figure_paths.append(snelson_figure)
    sgpr_tex = REPORT / "tables/kin40k_sgpr_hard_pic_512.tex"
    sgpr = sgpr_table(groups, sgpr_tex)
    source_hashes = {
        name: {key: value[key] for key in
               ("aggregate_sha256", "certificate_sha256") if key in value}
        for name, value in frozen_provenance.items()
    }
    manifest = args.output / "HARD_PIC_PAPER_ASSETS.json"
    manifest.write_text(json.dumps({
        "implementation": "hard-pic-pitc-paper-assets-v1",
        "scope": "3880 paired hard-block predictions; frozen training untouched",
        "hard_prediction_provenance": prediction_provenance,
        "frozen_campaign_provenance": source_hashes,
        "sgpr": sgpr,
        "snelson_hard_record_sha256": digest(snelson / "snelson_hard_prediction.json"),
        "outputs": {str(path.relative_to(PROJECT)): digest(path)
                    for path in [summary, sgpr_tex, *figure_paths]},
        "caveat": "Finite block-diagonal conditional predictor, not the exact dense variational covariance.",
    }, indent=2, sort_keys=True) + "\n")
    print(f"CERTIFIED={prediction_provenance['prediction_files']} "
          f"FIGURES={len(figure_paths)} MANIFEST={manifest}")


if __name__ == "__main__":
    main()
