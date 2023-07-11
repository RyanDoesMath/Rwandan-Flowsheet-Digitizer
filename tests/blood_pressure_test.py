"""Unit test file for blood_pressure.py"""

import unittest
import blood_pressure


class TestTimestampImputation(unittest.TestCase):
    """Tests the imputation of timestamps."""

    def test_transpose_dists_empty_list(self):
        """Tests transpose_dists with an empty list."""
        self.assertEqual([], blood_pressure.transpose_dists([]))


if __name__ == "__main__":
    unittest.main()
