#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``requirements.txt`` and the release profile cannot drift apart.

The runners refuse a runtime that does not match
``common.provenance.RELEASE_PACKAGE_VERSIONS``, and a clean environment is
built from ``requirements.txt``. Those were two hand-maintained copies of the
same numbers, so a bumped requirement could silently make every run
unreproducible-by-the-checker, or the reverse.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from common import provenance

SIM_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = SIM_ROOT / "requirements.txt"

_PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(?P<version>[^\s;]+)"
)


#: ``name @ git+<url>@<40-hex commit>`` -- a PEP 508 direct reference. A commit
#: is as exact as a version pin, and it is the only way to pin a distribution
#: that is not on PyPI. ``uci-datasets`` is one: the bare ``==1.1.0`` pin it
#: used to carry could never resolve, so the file was not installable at all.
_DIRECT = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*@\s*\S+@(?P<commit>[0-9a-f]{40})\s*$"
)
_DIRECT_COMMIT = re.compile(r"[0-9a-f]{40}")


def parse_requirements(text):
    """Return ``{distribution: pin}`` for every exactly-pinned requirement.

    A pin is a ``==`` version or the commit of a direct reference; both name one
    immutable artefact, which is what the release profile needs.
    """
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
            continue
        match = _DIRECT.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("commit")
    return pins


class ReleasePinsMatchRequirements(unittest.TestCase):
    def setUp(self):
        self.pins = parse_requirements(REQUIREMENTS.read_text())

    def test_requirements_file_is_fully_pinned(self):
        lines = [
            line.strip() for line in REQUIREMENTS.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # Exact means one immutable artefact: a "==" version, or the commit of
        # a direct reference for a distribution that is not on PyPI.
        unpinned = [
            line for line in lines
            if "==" not in line and not _DIRECT.match(line)
        ]
        self.assertEqual(unpinned, [], f"unpinned requirements: {unpinned}")

    def test_a_direct_reference_counts_as_a_pin(self):
        pins = parse_requirements(
            "foo==1.2.3\n"
            "bar @ git+https://example.invalid/bar.git@" + "a" * 40 + "\n"
            "# comment\n"
        )
        self.assertEqual(pins, {"foo": "1.2.3", "bar": "a" * 40})

    def test_an_unpinned_direct_reference_is_not_a_pin(self):
        self.assertEqual(
            parse_requirements("bar @ git+https://example.invalid/bar.git"), {}
        )

    def test_every_release_package_is_pinned_to_the_declared_version(self):
        mismatched = {}
        for name, version in provenance.RELEASE_PACKAGE_VERSIONS.items():
            pinned = self.pins.get(name.lower())
            # The CUDA plugin wheels are pulled in by the jax[cuda12] extra and
            # are not named in the file; they are checked at run time instead.
            if pinned is None and name.startswith("jax-cuda12"):
                continue
            # A direct reference pins a commit; the profile carries the version
            # importlib.metadata reports for that commit, so the two are exact
            # but not comparable as strings.
            if pinned is not None and _DIRECT_COMMIT.fullmatch(pinned):
                continue
            if pinned != version:
                mismatched[name] = (version, pinned)
        self.assertEqual(mismatched, {},
                         f"release profile disagrees with requirements.txt: {mismatched}")

    def test_every_release_critical_requirement_appears_in_the_profile(self):
        # A package a stack declares as release-critical must have a declared
        # version, otherwise release_runtime_matches would raise a KeyError at
        # the worst possible moment: while validating a finished campaign.
        critical = set(provenance.SVGP_RELEASE_PACKAGES) \
            | set(provenance.SGPR_RELEASE_PACKAGES) \
            | set(provenance.SNELSON_RELEASE_PACKAGES)
        missing = sorted(critical - set(provenance.RELEASE_PACKAGE_VERSIONS))
        self.assertEqual(missing, [])

    def test_python_version_file_matches_the_release_minor(self):
        version_file = SIM_ROOT / ".python-version"
        if not version_file.exists():
            self.skipTest(".python-version is not present")
        declared = version_file.read_text().strip()
        self.assertTrue(
            declared.startswith(provenance.RELEASE_PYTHON_MINOR),
            f".python-version says {declared}, release profile says "
            f"{provenance.RELEASE_PYTHON_MINOR}",
        )

class ImplementationIdsHaveOneOwner(unittest.TestCase):
    """An implementation id is declared once and imported everywhere else.

    Four modules used to carry their own literal copy. Bumping the benchmark
    left the table generator and the ablation plotter still filtering for the
    previous id, so they would have silently accepted zero runs -- the failure
    mode is an empty table, not an error.
    """

    def test_only_the_producing_module_declares_an_id(self):
        import re

        # The two campaign ids have one owner each. Exploratory studies under
        # */analysis/ are separate experiments with their own identity, not
        # consumers of a campaign id, so each may declare its own -- but it must
        # not collide with a campaign id.
        owners, exploratory = [], []
        for path in sorted(SIM_ROOT.glob("*/**/*.py")):
            if "_replaced" in str(path):
                continue
            for line in path.read_text().splitlines():
                match = re.match(r'^IMPLEMENTATION = "([^"]+)"', line)
                if not match:
                    continue
                name = path.relative_to(SIM_ROOT).as_posix()
                # The prepared-prediction backend versions its own code path
                # and never labels campaign rows.
                separate = "/analysis/" in f"/{name}" or name == "svgp/fast_prediction.py"
                (exploratory if separate else owners).append(
                    (name, match.group(1))
                )
        self.assertEqual(
            sorted(name for name, _ in owners),
            ["sgpr/benchmark.py", "svgp/engine.py"],
        )
        campaign_ids = {value for _, value in owners}
        for name, value in exploratory:
            self.assertNotIn(value, campaign_ids, name)
            # An exploratory id must not look like a campaign id, or a table
            # filter could accept rows from the wrong experiment.
            self.assertFalse(
                value.startswith(("sgpr-collapsed-", "svgp-residual-")), name
            )

    def test_consumers_agree_with_the_owner(self):
        from sgpr import benchmark
        from sgpr import make_main_table
        from sgpr import plot_predictive_nll_ablation
        from svgp import engine
        from svgp.analysis import paper_status

        self.assertEqual(make_main_table.IMPLEMENTATION, benchmark.IMPLEMENTATION)
        self.assertEqual(
            plot_predictive_nll_ablation.IMPLEMENTATION, benchmark.IMPLEMENTATION
        )
        self.assertEqual(paper_status.IMPLEMENTATION, engine.IMPLEMENTATION)


if __name__ == "__main__":
    unittest.main()
