"""Certified control aggregation keeps paired references immutable."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import collect_partition_controls as collector
import real_predictor_campaign as real


class CollectControlTests(unittest.TestCase):
    def test_paired_differences_and_reference_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            root=base/"E1"
            reference=base/"reference.json"
            real.atomic_json(reference,dict(nll=.2))
            ref_hash=real.sha256(reference)
            row=dict(kernel="rbf",dataset="pol",M=64,method="RBT-SVGP",seed=0,
                     family="svgp",output_relative="svgp/rbf/pol/M64/RBT-SVGP/seed0",
                     reference=str(reference),reference_sha256=ref_hash,
                     root=str(base/"frozen"),relative="run",frozen_inputs={})
            real.atomic_json(base/"frozen"/"run"/"training.json",dict(config_id="fixed",config=dict(block=500)))
            real.atomic_json(root/"plan.json",dict(experiment="E1",runs=[row],control_options={}))
            path=root/"hard"/row["output_relative"]/"prediction.json"
            real.atomic_json(path,dict(valid=True,rmse=.3,nll=.1,identity=dict(config_id="fixed",
                inputs={},plan_sha256=real.sha256(root/"plan.json"))))
            real.atomic_json(root/"PARTITION_CONTROL_CERTIFICATE.json",dict(complete=True,
                plan_sha256=real.sha256(root/"plan.json"),predictions={row["output_relative"]:
                dict(sha256=real.sha256(path))}))
            collector.collect([root],base/"collected")
            with (base/"collected"/"paired_per_seed.csv").open() as stream:
                record=next(csv.DictReader(stream))
            self.assertAlmostEqual(float(record["nll_difference_from_current"]),-.1)
            self.assertEqual(real.sha256(reference),ref_hash)
            with self.assertRaises(ValueError):
                collector.collect([root],base/"collected")
            real.atomic_json(path,dict(valid=True,rmse=.3,nll=.5))
            with self.assertRaises(ValueError):
                collector.collect([root],base/"tampered")


if __name__ == "__main__":
    unittest.main()
