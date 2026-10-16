"""Protocol/dispatch tests require neither JAX nor a GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "gibbs_anchor_campaign.py"
SPEC = importlib.util.spec_from_file_location("anchor_campaign", PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class AnchorCampaignTests(unittest.TestCase):
    def reference(self):
        return dict(dataset="pol", M=128, seed=3, split_seed=3, epochs=300,
                    block=500, ll_batch_size=1000, blocks_per_step=2,
                    refresh_every=40, refresh_until=81, batch_coupling="split",
                    penalty_sampling="uniform", optimizer="adam", lr=.005,
                    z_lr=.005, lr_profile="fixed", lr_schedule="constant",
                    q_optimizer="natural", natural_lr=.05, warp_gate_fraction=0.,
                    swa_fraction=0., grad_clip=10., residual_probes=32,
                    importance_refresh_every=10, importance_epsilon=.05,
                    log_every=20, split_rng="numpy-pcg64-v1", full_predictor_max_n=50000,
                    sbt_geometry="residual-probe", run_tag="old", plan="grid-exploratory")

    def test_only_geometry_and_identity_metadata_change(self):
        original = self.reference()
        new = campaign.expected_config(original)
        self.assertEqual({k for k in new if new[k] != original[k]},
                         {"sbt_geometry", "plan", "run_tag"})
        self.assertEqual(original["sbt_geometry"], "residual-probe")

    def test_cli_explicit_anchor_no_force_no_resume(self):
        task = dict(config=campaign.expected_config(self.reference()), worker=3)
        cmd = campaign.command(dict(python="/python", source="/source"), Path("/new"), task)
        self.assertEqual(cmd[cmd.index("--sbt-geometry") + 1], "anchor")
        self.assertEqual(cmd[cmd.index("--init-from-runs-dir") + 1], "/new/inputs")
        self.assertIn("--defer-prediction", cmd)
        self.assertNotIn("--force", cmd)
        self.assertNotIn("--resume-from-kernel", cmd)
        # Every argument exists in the source CLI; no guessed launcher flags.
        tree = ast.parse((PATH.parent / "svgp/run_benchmark.py").read_text())
        options = {a.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                   for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        # --residual-probes belongs to the pinned source tree the campaign
        # executes (SOURCE_HEAD); the current CLI no longer declares it.
        pinned_only = {"--residual-probes"}
        self.assertFalse({a for a in cmd if a.startswith("--")} - options - pinned_only)

    def test_five_disjoint_balanced_shards(self):
        shards = [set() for _ in range(5)]
        for d in campaign.DATASETS:
            for m in (64, 128, 256, 512):
                for seed in range(10):
                    shards[seed % 5].add((d, m, seed))
        self.assertEqual([len(x) for x in shards], [64] * 5)
        self.assertEqual(len(set.union(*shards)), 320)
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(shards) for b in shards[i+1:]))
        self.assertIn(("3droad", 512, 9), set.union(*shards))

    def test_protected_directory_overlap(self):
        for a, b in (("/old", "/old/new"), ("/old", "/old"), ("/old/new", "/old")):
            with self.assertRaises(ValueError):
                campaign.disjoint(a, b)
        campaign.disjoint("/old", "/new")

    def test_residual_probe_cannot_be_accepted_as_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "training.json").write_text(json.dumps(dict(config=self.reference())))
            task = dict(run="run", config=campaign.expected_config(self.reference()))
            self.assertTrue(campaign.validate_task(root, task).startswith("invalid:"))

    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "campaign.json").write_text("{}")
            (root / "campaign.sha256").write_text("different")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                campaign.load_plan(root)


if __name__ == "__main__":
    unittest.main()
