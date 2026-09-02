import argparse
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from cgra_ii_predictor.cli import _numbers, main


class CliTest(unittest.TestCase):
    def test_candidate_parser_rejects_non_finite_invalid_and_empty_values(self):
        for value in ("nan", "inf", "-inf", "0", "-1", ""):
            with self.subTest(ridge=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                _numbers(value, allow_zero=False)
        for value in ("nan", "inf", "-inf", "-1", ","):
            with self.subTest(dead_zone=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                _numbers(value, allow_zero=True)

    def test_main_turns_invalid_candidate_grid_into_system_exit(self):
        for option, value in (
            ("--ridge-candidates", "nan"),
            ("--dead-zone-candidates", ""),
        ):
            with self.subTest(option=option):
                error = io.StringIO()
                with patch.object(
                    sys, "argv", ["cgra-ii-predictor", "unused.json", option, value]
                ), contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(option, error.getvalue())


if __name__ == "__main__":
    unittest.main()
