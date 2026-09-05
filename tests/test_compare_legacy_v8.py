import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class LegacyV8ComparisonTest(unittest.TestCase):
    def test_loads_validation_selected_ensemble_and_infers_legacy_mode(self):
        from adapters.compare_legacy_v8_mapper4x4 import load_ensemble

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            source_manifest = root / "manifest.json"
            source_manifest.write_text(json.dumps({
                "candidates": [{"ranking_query_id": "q"}],
            }))
            report = root / "ensemble.json"
            report.write_text(json.dumps({
                "selection_split": "validation_only",
                "manifest": str(source_manifest),
                "manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "checkpoints": {
                    "model": {"path": str(checkpoint), "sha256": digest},
                },
                "weights": {
                    "analytical_lower_bound": 0.25,
                    "model": 0.75,
                },
                "uncertainty_gating": {
                    "exponent": 2.0,
                    "scales": {
                        "analytical_lower_bound": 1.0,
                        "model": 0.5,
                    },
                },
                "validation": {
                    "ensemble": {"mae": 0.3},
                    "static_ensemble": {"mae": 0.2},
                },
            }))
            loaded = load_ensemble(report)
            self.assertEqual(loaded["selected_mode"], "static")
            self.assertEqual(loaded["checkpoints"], {"model": checkpoint})
            self.assertEqual(loaded["source_manifest"], source_manifest)

    def test_rejects_non_validation_selection(self):
        from adapters.compare_legacy_v8_mapper4x4 import load_ensemble

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "ensemble.json"
            report.write_text(json.dumps({"selection_split": "test"}))
            with self.assertRaisesRegex(ValueError, "validation only"):
                load_ensemble(report)


if __name__ == "__main__":
    unittest.main()
