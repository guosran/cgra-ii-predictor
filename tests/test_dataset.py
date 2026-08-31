import json
import tempfile
import unittest
from pathlib import Path

from cgra_ii_predictor.dataset import load_dataset


class DatasetTest(unittest.TestCase):
    def test_loads_portable_schema(self):
        path = Path(__file__).parents[1] / "schema/example-dataset.json"
        dataset = load_dataset(path)
        self.assertEqual(len(dataset.samples), 3)
        self.assertEqual(dataset.feature_names, ("pressure", "semantic_depth"))

    def test_rejects_compiled_ii_below_bound(self):
        raw = {
            "feature_names": ["x"],
            "samples": [{
                "sample_id": "bad", "group": "bad", "lower_bound": 4,
                "compiled_ii": 3, "features": {"x": 1},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()

