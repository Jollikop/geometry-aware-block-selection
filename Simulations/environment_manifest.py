#!/usr/bin/env python3
"""Write the runtime and source state used by an experiment release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.provenance import code_state, runtime_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    payload = {
        "code": code_state(project_root),
        "runtime": runtime_manifest(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
