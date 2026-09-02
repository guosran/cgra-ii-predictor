import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from adapters import neura_experiment as adapter
from adapters import neura_motifs


COST_TEXT = (
    "rec_res_mii_info = {rec_mii = 1 : i32 res_mii = 2 : i32}"
)
MAPPED_TEXT = (
    'mapping_info = {mapping_strategy = "heuristic"} '
    "compiled_ii = 3 : i32 rec_mii = 1 : i32 res_mii = 2 : i32"
)


class MotifCollectionTest(unittest.TestCase):
    def setUp(self):
        adapter.INVOCATION_FAILURES.clear()

    def tearDown(self):
        adapter.INVOCATION_FAILURES.clear()

    def fresh_corpus(
        self, root, *, count=1, shapes=((3, 3),),
        variants=("homogeneous",), jobs=2,
    ):
        opt = root / "mlir-neura-opt"
        opt.write_bytes(b"test-opt-v1")
        manifest_path = root / "corpus-manifest.json"
        prepared = adapter._load_or_create_motif_manifest(
            root, manifest_path,
            resume=False,
            clean=False,
            count=count,
            seed=17,
            motifs=("chain",),
            shapes=shapes,
            variants=variants,
            timeout=5,
            jobs=jobs,
            checkpoint_every=1,
            opt=opt,
        )
        return opt, manifest_path, prepared

    @staticmethod
    def successful_result(candidate, delay=0.0):
        if delay:
            time.sleep(delay)
        sample_dir = Path(candidate.source_path).parent
        cost = sample_dir / "cost.mlir"
        mapped = sample_dir / "mapped.mlir"
        cost.write_text(COST_TEXT)
        mapped.write_text(MAPPED_TEXT)
        values = adapter.parse_cost_features(COST_TEXT)
        sample = adapter._motif_sample_from_artifacts(
            candidate, values, 3, cost, mapped
        )
        return adapter.MotifCollectionResult(
            candidate.candidate_id,
            "success",
            "mapper",
            sample=sample,
        )

    def test_candidate_protocol_error_propagates_instead_of_becoming_censored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, _manifest_path, prepared = self.fresh_corpus(root)
            candidate = prepared[1][0]

            def mismatched_invocation(command, timeout):
                output = Path(command[-1])
                stage = adapter.invocation_stage(command)
                if stage == "rec-res-analysis":
                    output.write_text(COST_TEXT)
                else:
                    output.write_text(
                        "compiled_ii = 3 : i32 rec_mii = 9 : i32 "
                        "res_mii = 2 : i32"
                    )
                return adapter.InvocationResult(
                    True, "success", stage, timeout, tuple(command)
                )

            with self.assertRaisesRegex(ValueError, "facts disagree"):
                adapter.collect_motif_candidate(
                    opt, candidate, 5, mismatched_invocation
                )

    def test_changed_predeclared_input_is_protocol_error_before_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, _manifest_path, prepared = self.fresh_corpus(root)
            candidate = prepared[1][0]
            Path(candidate.source_path).write_text("module {}\n")
            invocation = mock.Mock(
                side_effect=AssertionError("compiler must not be invoked")
            )
            with self.assertRaisesRegex(ValueError, "input hash changed"):
                adapter.collect_motif_candidate(
                    opt, candidate, 5, invocation
                )
            invocation.assert_not_called()

    def test_parallel_collection_is_bounded_and_emitted_in_candidate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(
                root,
                shapes=((2, 2), (2, 3), (3, 3), (3, 4)),
                jobs=3,
            )
            manifest, candidates, cached, _declared, failures = prepared
            ordinal = {
                candidate.candidate_id: index
                for index, candidate in enumerate(candidates)
            }
            lock = threading.Lock()
            active = 0
            maximum_active = 0
            worker_threads = []

            def worker(_opt, candidate, _timeout, _invocation):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                    worker_threads.append(threading.current_thread().name)
                try:
                    delay = 0.004 * (len(candidates) - ordinal[candidate.candidate_id])
                    return self.successful_result(candidate, delay)
                finally:
                    with lock:
                        active -= 1

            atomic_threads = []
            real_atomic_write = neura_motifs.atomic_write_json

            def checked_atomic_write(path, payload):
                atomic_threads.append(threading.current_thread().name)
                real_atomic_write(path, payload)

            coordinator = adapter.MotifCollectionCoordinator(
                opt, candidates, manifest_path, manifest, 5, 3, 1,
                cached, failures,
            )
            with mock.patch.object(
                adapter, "collect_motif_candidate", side_effect=worker
            ), mock.patch.object(
                neura_motifs, "atomic_write_json", side_effect=checked_atomic_write
            ):
                result = coordinator.run()

            expected_ids = [candidate.candidate_id for candidate in candidates]
            self.assertEqual(
                [row["candidate_id"] for row in result.samples], expected_ids
            )
            self.assertEqual(
                [row.candidate_id for row in result.results], expected_ids
            )
            self.assertLessEqual(maximum_active, 3)
            self.assertGreater(maximum_active, 1)
            self.assertTrue(all(name.startswith("motif-worker")
                                for name in worker_threads))
            self.assertTrue(atomic_threads)
            self.assertEqual(set(atomic_threads), {"MainThread"})
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(
                [row["id"] for row in persisted["candidates"]], expected_ids
            )
            self.assertEqual(persisted["status"], "complete")
            self.assertFalse(any(
                row["status"] == "running" for row in persisted["candidates"]
            ))

    def test_worker_protocol_exception_keeps_all_candidates_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(
                root, shapes=((2, 2), (3, 3)), jobs=1
            )
            manifest, candidates, cached, _declared, failures = prepared
            coordinator = adapter.MotifCollectionCoordinator(
                opt, candidates, manifest_path, manifest, 5, 1, 1,
                cached, failures,
            )
            with mock.patch.object(
                adapter, "collect_motif_candidate",
                side_effect=ValueError("protocol mismatch"),
            ) as worker, self.assertRaisesRegex(RuntimeError, "worker failed"):
                coordinator.run()
            self.assertEqual(worker.call_count, 1)
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(persisted["status"], "interrupted")
            self.assertTrue(all(
                row["status"] == "declared" for row in persisted["candidates"]
            ))

    def test_terminal_resume_skips_success_and_censored_without_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(
                root, shapes=((2, 2), (3, 3)), jobs=2
            )
            manifest, candidates, cached, _declared, failures = prepared
            timeout_result = adapter.InvocationResult(
                False,
                "timeout",
                "mapper",
                5,
                (str(opt), "--map-to-accelerator"),
                output="mapped.mlir",
            )

            def worker(_opt, candidate, _timeout, _invocation):
                if candidate.candidate_id == candidates[0].candidate_id:
                    return self.successful_result(candidate)
                return adapter.MotifCollectionResult(
                    candidate.candidate_id,
                    "censored",
                    "mapper",
                    "timeout",
                    invocations=(timeout_result,),
                )

            with mock.patch.object(
                adapter, "collect_motif_candidate", side_effect=worker
            ):
                adapter.MotifCollectionCoordinator(
                    opt, candidates, manifest_path, manifest, 5, 2, 1,
                    cached, failures,
                ).run()

            opt.unlink()
            resumed = adapter._load_or_create_motif_manifest(
                root, manifest_path,
                resume=True,
                clean=False,
                count=1,
                seed=17,
                motifs=("chain",),
                shapes=((2, 2), (3, 3)),
                variants=("homogeneous",),
                timeout=5,
                jobs=4,
                checkpoint_every=7,
                opt=root / "compiler-is-unavailable",
            )
            resumed_manifest, resumed_candidates, resumed_samples, declared, prior = resumed
            self.assertEqual(declared, [])
            self.assertEqual(set(resumed_samples), {candidates[0].candidate_id})
            self.assertEqual(
                prior[candidates[1].candidate_id][0]["status"], "timeout"
            )
            adapter.INVOCATION_FAILURES.clear()
            coordinator = adapter.MotifCollectionCoordinator(
                root / "compiler-is-unavailable",
                resumed_candidates,
                manifest_path,
                resumed_manifest,
                5,
                4,
                7,
                resumed_samples,
                prior,
            )
            with mock.patch.object(
                adapter, "collect_motif_candidate",
                side_effect=AssertionError("terminal candidates were retried"),
            ) as worker:
                result = coordinator.run()
            worker.assert_not_called()
            self.assertFalse(result.interrupted)
            self.assertEqual(len(result.samples), 1)
            self.assertEqual(
                [row["candidate_id"] for row in adapter.INVOCATION_FAILURES],
                [candidates[1].candidate_id],
            )

    def test_resume_rejects_changed_compiler_and_corrupt_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(root)
            manifest, candidates, cached, _declared, failures = prepared
            opt.write_bytes(b"different-compiler")
            with self.assertRaisesRegex(ValueError, "compiler SHA-256"):
                adapter._load_or_create_motif_manifest(
                    root, manifest_path,
                    resume=True,
                    clean=False,
                    count=1,
                    seed=17,
                    motifs=("chain",),
                    shapes=((3, 3),),
                    variants=("homogeneous",),
                    timeout=5,
                    jobs=1,
                    checkpoint_every=1,
                    opt=opt,
                )

            opt.write_bytes(b"test-opt-v1")
            with mock.patch.object(
                adapter, "collect_motif_candidate",
                side_effect=lambda _opt, candidate, _timeout, _invocation:
                    self.successful_result(candidate),
            ):
                adapter.MotifCollectionCoordinator(
                    opt, candidates, manifest_path, manifest, 5, 1, 1,
                    cached, failures,
                ).run()
            mapped = Path(candidates[0].source_path).parent / "mapped.mlir"
            mapped.write_text(MAPPED_TEXT + " tampered")
            with self.assertRaisesRegex(ValueError, "mapped artifact hash mismatch"):
                adapter._load_or_create_motif_manifest(
                    root, manifest_path,
                    resume=True,
                    clean=False,
                    count=1,
                    seed=17,
                    motifs=("chain",),
                    shapes=((3, 3),),
                    variants=("homogeneous",),
                    timeout=5,
                    jobs=1,
                    checkpoint_every=1,
                    opt=opt,
                )

    def test_partial_resume_records_relocated_identical_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, _prepared = self.fresh_corpus(root)
            relocated_opt = root / "relocated-mlir-neura-opt"
            relocated_opt.write_bytes(opt.read_bytes())

            resumed = adapter._load_or_create_motif_manifest(
                root, manifest_path,
                resume=True,
                clean=False,
                count=1,
                seed=17,
                motifs=("chain",),
                shapes=((3, 3),),
                variants=("homogeneous",),
                timeout=5,
                jobs=1,
                checkpoint_every=1,
                opt=relocated_opt,
            )

            resumed_manifest, _candidates, _samples, declared, _prior = resumed
            self.assertTrue(declared)
            self.assertEqual(
                resumed_manifest["collection"]["mlir_neura_opt"],
                str(relocated_opt.resolve()),
            )
            self.assertEqual(
                resumed_manifest["collection"]["mlir_neura_opt_sha256"],
                adapter.file_sha256(relocated_opt),
            )
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(
                persisted["collection"]["mlir_neura_opt"],
                str(relocated_opt.resolve()),
            )

    def test_cooperative_stop_drains_only_bounded_inflight_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(
                root,
                shapes=((2, 2), (2, 3), (3, 3), (3, 4), (4, 4)),
                jobs=2,
            )
            manifest, candidates, cached, _declared, failures = prepared
            started = threading.Event()
            release = threading.Event()
            lock = threading.Lock()
            calls = []

            def worker(_opt, candidate, _timeout, _invocation):
                with lock:
                    calls.append(candidate.candidate_id)
                    if len(calls) == 2:
                        started.set()
                if not release.wait(2):
                    raise RuntimeError("test worker was not released")
                return self.successful_result(candidate)

            coordinator = adapter.MotifCollectionCoordinator(
                opt, candidates, manifest_path, manifest, 5, 2, 1,
                cached, failures,
            )
            holder = {}

            def run_coordinator():
                holder["result"] = coordinator.run()

            with mock.patch.object(
                adapter, "collect_motif_candidate", side_effect=worker
            ):
                thread = threading.Thread(target=run_coordinator)
                thread.start()
                self.assertTrue(started.wait(2))
                coordinator.request_stop()
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(calls), 2)
            self.assertTrue(holder["result"].interrupted)
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(persisted["status"], "interrupted")
            self.assertEqual(persisted["summary"]["success_count"], 2)
            self.assertEqual(
                persisted["summary"]["declared_count"], len(candidates) - 2
            )

    def test_cli_rejects_invalid_parallel_collection_options(self):
        cases = (
            ("--motif-jobs", "0"),
            ("--motif-checkpoint-every", "0"),
        )
        for option, value in cases:
            with self.subTest(option=option), mock.patch.object(
                sys, "argv", ["neura_experiment.py", option, value]
            ), self.assertRaises(SystemExit) as raised:
                adapter.parse_args()
            self.assertEqual(raised.exception.code, 2)
        with mock.patch.object(
            sys,
            "argv",
            ["neura_experiment.py", "--clean", "--motif-resume"],
        ), self.assertRaises(SystemExit) as raised:
            adapter.parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_terminal_main_resume_does_not_probe_or_invoke_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt, manifest_path, prepared = self.fresh_corpus(
                root,
                count=2,
                shapes=((3, 3), (3, 4), (4, 4)),
                variants=("homogeneous", "split-domain"),
                jobs=4,
            )
            manifest, candidates, cached, _declared, failures = prepared
            with mock.patch.object(
                adapter, "collect_motif_candidate",
                side_effect=lambda _opt, candidate, _timeout, _invocation:
                    self.successful_result(candidate),
            ):
                adapter.MotifCollectionCoordinator(
                    opt, candidates, manifest_path, manifest, 5, 4, 4,
                    cached, failures,
                ).run()
            opt.unlink()
            architecture = root / "unused-architecture.yaml"
            architecture.write_text("unused: true\n")
            missing_opt = root / "missing-compiler"
            argv = [
                "neura_experiment.py",
                "--motif-resume",
                "--output-dir", str(root),
                "--opt", str(missing_opt),
                "--real-architecture", str(architecture),
                "--metadata-holdout-key", "generator_family",
                "--timeout", "5",
            ]
            adapter.INVOCATION_FAILURES.clear()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                adapter, "require_opt_argument",
                side_effect=AssertionError("compiler help was probed"),
            ) as probe, mock.patch.object(
                adapter, "run_invocation",
                side_effect=AssertionError("compiler was invoked"),
            ) as isolated, mock.patch.object(
                adapter, "invoke",
                side_effect=AssertionError("legacy compiler path was invoked"),
            ) as legacy:
                self.assertEqual(adapter.main(), 0)
            probe.assert_not_called()
            isolated.assert_not_called()
            legacy.assert_not_called()
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(
                report["provenance"]["mlir_neura_opt_sha256"],
                manifest["collection"]["mlir_neura_opt_sha256"],
            )
            self.assertTrue(
                report["provenance"]["experiment_config"]["motif_resume"]
            )
            self.assertNotEqual(
                report["nested_ridge_metadata_holdouts"][
                    "generator_family"
                ]["status"],
                "not_requested",
            )
            self.assertEqual(
                report["candidate_gate"]["requested_bases_per_family"], 2
            )
            self.assertEqual(
                report["candidate_gate"]["coverage"][
                    "requested_bases_per_family"
                ],
                2,
            )

            # Reusing the directory for an ordinary input-report run must not
            # attribute its stale corpus manifest to the new invocation.
            input_report = root / "resumed-report.json"
            input_report.write_text(json.dumps(report))
            argv = [
                "neura_experiment.py",
                "--input-report", str(input_report),
                "--output-dir", str(root),
                "--opt", str(missing_opt),
                "--real-architecture", str(architecture),
                "--timeout", "5",
            ]
            adapter.INVOCATION_FAILURES.clear()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                adapter, "require_opt_argument",
                side_effect=AssertionError("compiler help was probed"),
            ) as probe, mock.patch.object(
                adapter, "run_invocation",
                side_effect=AssertionError("compiler was invoked"),
            ) as isolated, mock.patch.object(
                adapter, "invoke",
                side_effect=AssertionError("legacy compiler path was invoked"),
            ) as legacy:
                self.assertEqual(adapter.main(), 0)
            probe.assert_not_called()
            isolated.assert_not_called()
            legacy.assert_not_called()
            ordinary_report = json.loads((root / "report.json").read_text())
            self.assertIsNone(
                ordinary_report["motif_corpus"]["manifest_path"]
            )
            self.assertIsNone(
                ordinary_report["motif_corpus"]["manifest_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
