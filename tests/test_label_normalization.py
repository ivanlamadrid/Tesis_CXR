"""Tests for NIH label normalization."""

from __future__ import annotations

import unittest

from src.utils.labels import normalize_finding_label


class LabelNormalizationTest(unittest.TestCase):
    def test_pleural_thickening_alias(self) -> None:
        self.assertEqual(
            normalize_finding_label("Pleural_Thickening"),
            "Pleural Thickening",
        )

    def test_pleural_thickening_alias_with_spaces(self) -> None:
        self.assertEqual(
            normalize_finding_label(" Pleural_Thickening "),
            "Pleural Thickening",
        )

    def test_no_finding_is_preserved(self) -> None:
        self.assertEqual(normalize_finding_label("No Finding"), "No Finding")


if __name__ == "__main__":
    unittest.main()
