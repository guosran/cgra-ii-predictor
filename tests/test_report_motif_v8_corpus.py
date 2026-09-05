import json
from pathlib import Path
import tempfile
import unittest


class MotifV8CorpusReportTest(unittest.TestCase):
    def test_rejects_a_nonterminal_manifest(self):
        from adapters.report_motif_v8_corpus import build_report

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "corpus-manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "cgra-ii-motif-corpus-v8",
                "status": "running",
                "shape_protocol": {
                    "protocol_id": "amoeba-static-rectangles-4x4-tiles-v1",
                },
                "candidates": [{"status": "declared"}],
            }))
            with self.assertRaisesRegex(ValueError, "fully terminal"):
                build_report(manifest)


if __name__ == "__main__":
    unittest.main()
