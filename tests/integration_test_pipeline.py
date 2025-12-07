import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import yaml

from csrlite.ae.ae_summary import study_plan_to_ae_summary
from csrlite.common.plan import load_plan


class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Create Dummy Data
            adsl = pl.DataFrame(
                {
                    "USUBJID": ["sub1", "sub2", "sub3"],
                    "TRT01P": ["Active", "Active", "Placebo"],
                    "SAFFL": ["Y", "Y", "Y"],
                }
            )
            adae = pl.DataFrame(
                {"USUBJID": ["sub1", "sub2"], "AESER": ["Y", "N"], "TRT01P": ["Active", "Active"]}
            )

            adsl_path = tmp_path / "adsl.parquet"
            adae_path = tmp_path / "adae.parquet"
            adsl.write_parquet(adsl_path)
            adae.write_parquet(adae_path)

            # 2. Create YAML Plan
            plan_data = {
                "study": {"name": "Integration Test", "output": str(tmp_path / "output")},
                "data": [
                    {"name": "adsl", "path": "adsl.parquet"},
                    {"name": "adae", "path": "adae.parquet"},
                ],
                "population": [
                    {"name": "safety", "label": "Safety Population", "filter": "SAFFL == 'Y'"}
                ],
                "observation": [
                    # Observation types if needed, can be empty or default
                ],
                "parameter": [{"name": "any_ae", "label": "Any AE", "filter": "True"}],
                "group": [
                    {"name": "treatment", "variable": "TRT01P", "label": ["Active", "Placebo"]}
                ],
                "plans": [
                    {
                        "analysis": "ae_summary",
                        "population": "safety",
                        "parameter": "any_ae",
                        "group": "treatment",
                    }
                ],
            }

            plan_file = tmp_path / "plan.yaml"
            with open(plan_file, "w") as f:
                yaml.dump(plan_data, f)

            # 3. Load Plan
            study_plan = load_plan(str(plan_file))
            self.assertIsNotNone(study_plan)
            self.assertIn("adsl", study_plan.datasets)

            # 4. Run Analysis
            outputs = study_plan_to_ae_summary(study_plan)

            # 5. Verify Output
            self.assertTrue(len(outputs) > 0)
            output_file = Path(outputs[0])
            self.assertTrue(output_file.exists())

            # Optionally check file size or content basics
            self.assertGreater(output_file.stat().st_size, 0)
