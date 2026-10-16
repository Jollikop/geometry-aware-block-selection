#!/usr/bin/env python3
"""Run the frozen SVGP program with a bounded, provenance-recorded predictor.

The historical campaign intentionally owns the numerical configuration and its
cache fingerprint.  Prediction chunking changes only how test rows are batched;
it does not change the predictor or the training protocol.  This wrapper keeps
the historical runner and modules in charge, applies one common chunk to every
residual predictor, and records the execution detail on newly written metrics.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import runpy
import sys
from pathlib import Path


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-runner", required=True)
    parser.add_argument("--prediction-chunk", required=True, type=int)
    parser.add_argument("--orchestrator-commit", required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = args.runner_args
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    if args.prediction_chunk < 1:
        parser.error("--prediction-chunk must be positive")
    return args, runner_args


def main(argv: list[str] | None = None) -> None:
    args, runner_args = parse_wrapper_args(list(sys.argv[1:] if argv is None else argv))
    source_runner = Path(args.source_runner).resolve()
    source_sim = source_runner.parent.parent
    if not source_runner.is_file():
        raise SystemExit(f"missing historical runner: {source_runner}")

    # The historical common/ and svgp/ modules must win over the wrapper's own
    # checkout.  This preserves the training code and its cached fingerprints.
    sys.path.insert(0, str(source_sim))
    from common import predictors  # pylint: disable=import-outside-toplevel

    original_predictive_variance = predictors.predictive_variance

    @functools.wraps(original_predictive_variance)
    def chunked_predictive_variance(*call_args, **call_kwargs):
        if call_kwargs.get("chunk") is None:
            call_kwargs["chunk"] = args.prediction_chunk
        return original_predictive_variance(*call_args, **call_kwargs)

    predictors.predictive_variance = chunked_predictive_variance
    os.environ["SVGP_PREDICTION_CHUNK"] = str(args.prediction_chunk)
    os.environ["SVGP_ORCHESTRATOR_COMMIT"] = str(args.orchestrator_commit)

    # Only files created by this process pass through this hook. Cached metrics
    # return before writing and therefore remain byte-for-byte untouched.
    original_write_text = Path.write_text

    def provenance_write_text(path: Path, data: str, *write_args, **write_kwargs):
        if path.name == "metrics.json":
            try:
                payload = json.loads(data)
            except (TypeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and str(
                payload.get("implementation", "")
            ).startswith("svgp-"):
                payload["prediction_execution"] = {
                    "test_chunk": args.prediction_chunk,
                    "policy": "same-test-chunk-for-every-residual-predictor",
                    "mathematical_predictor_unchanged": True,
                    "orchestrator_commit": args.orchestrator_commit,
                }
                data = json.dumps(payload, indent=2)
        return original_write_text(path, data, *write_args, **write_kwargs)

    Path.write_text = provenance_write_text
    print(
        "PREDICTION EXECUTION: common test chunk="
        f"{args.prediction_chunk} orchestrator={args.orchestrator_commit}"
    )
    sys.argv = [str(source_runner), *runner_args]
    runpy.run_path(str(source_runner), run_name="__main__")


if __name__ == "__main__":
    main()
