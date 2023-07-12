"""Unit test file for blood_pressure.py"""
import sys

sys.path.append("src")
import unittest
import blood_pressure


class TestTimestampImputation(unittest.TestCase):
    """Tests the imputation of timestamps."""

    def test_transpose_dists_empty_list(self):
        """Tests transpose_dists with an empty list."""
        self.assertEqual(blood_pressure.transpose_dists([]), [])

    def test_transpose_dists_one_dimensional(self):
        """Tests transpose_dists with a one dimensional list."""
        fn_input = [[0], [1], [2]]
        fn_output = [[0, 1, 2]]
        self.assertEqual(blood_pressure.transpose_dists(fn_input), fn_output)
        self.assertEqual(blood_pressure.transpose_dists(fn_output), fn_input)

    def test_transpose_dists_two_dimensional_square(self):
        """Tests transpose_dists with a square two dimensional list."""
        fn_input = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        fn_output = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        self.assertEqual(blood_pressure.transpose_dists(fn_input), fn_output)
        self.assertEqual(blood_pressure.transpose_dists(fn_output), fn_input)

    def test_transpose_dists_two_dimensional_non_square(self):
        """Tests transpose_dists with an arbitrary two dimensional list."""
        fn_input = [[0, 1], [3, 4], [6, 7]]
        fn_output = [[0, 3, 6], [1, 4, 7]]
        self.assertEqual(blood_pressure.transpose_dists(fn_input), fn_output)
        self.assertEqual(blood_pressure.transpose_dists(fn_output), fn_input)

    def test_get_index_of_smallest_val(self):
        """Tests the get_index_of_smallest_val function."""
        self.assertEqual(blood_pressure.get_index_of_smallest_val([1, 0, 2]), 1)
        self.assertEqual(blood_pressure.get_index_of_smallest_val([0, 1, 2]), 0)
        self.assertEqual(blood_pressure.get_index_of_smallest_val([1, 2, 0]), 2)
        self.assertEqual(blood_pressure.get_index_of_smallest_val([2, 0, 0]), 1)

    def test_get_index_of_list_with_smallest_min_val(self):
        """Tests the get_index_of_list_with_smallest_min_val function."""
        self.assertEqual(
            blood_pressure.get_index_of_list_with_smallest_min_val(
                [[3, 6], [1, 7], [0, 9]]
            ),
            2,
        )
        self.assertEqual(
            blood_pressure.get_index_of_list_with_smallest_min_val(
                [[3, 6], [1, 7], [1, 7]]
            ),
            1,
        )
        self.assertEqual(
            blood_pressure.get_index_of_list_with_smallest_min_val(
                [[3, -1], [1, 7], [0, 9]]
            ),
            0,
        )

    def test_get_index_of_list_with_largest_min_val(self):
        """Tests the get_index_of_list_with_largest_min_val function."""
        self.assertEqual(
            blood_pressure.get_index_of_list_with_largest_min_val(
                [[3, 6], [1, 7], [0, 9]]
            ),
            0,
        )
        self.assertEqual(
            blood_pressure.get_index_of_list_with_largest_min_val(
                [[1, 7], [3, 6], [1, 7]]
            ),
            1,
        )
        self.assertEqual(
            blood_pressure.get_index_of_list_with_largest_min_val(
                [[3, -1], [1, 7], [8, 9]]
            ),
            2,
        )

    def test_generate_x_dists_matrix(self):
        """Tests the generate_x_dists_matrix function."""
        fn_input = {
            "systolic": [[0, 0, 1, 1], [1, 1, 2, 2]],
            "diastolic": [[0, 0, 1, 1], [1, 1, 2, 2]],
        }
        self.assertEqual(
            blood_pressure.generate_x_dists_matrix(fn_input), [[0, 1], [1, 0]]
        )
        fn_input = {
            "systolic": [[0, 0, 1, 1], [1, 1, 2, 2]],
            "diastolic": [[1, 1, 2, 2], [2, 2, 3, 3]],
        }
        self.assertEqual(
            blood_pressure.generate_x_dists_matrix(fn_input), [[1, 2], [0, 1]]
        )
        fn_input = {
            "systolic": [[0, 0, 1, 1], [1, 0, 2, 1]],
            "diastolic": [[0, 1, 1, 2]],
        }
        self.assertEqual(blood_pressure.generate_x_dists_matrix(fn_input), [[0], [1]])

    def test_filter_non_matches(self):
        """Tests the filter_non_matches function."""
        # standard case.
        dist_input = [[0], [1]]
        bp_bounding_boxes_input = {
            "systolic": [[0, 0, 1, 1], [1, 0, 2, 1]],
            "diastolic": [[0, 1, 1, 2]],
        }
        true_output = (
            [[0]],
            [blood_pressure.BloodPressure([1, 0, 2, 1], None, -1, -1, -1)],
        )
        self.assertEqual(
            blood_pressure.filter_non_matches(dist_input, bp_bounding_boxes_input),
            true_output,
        )

        # no non matches.
        dist_input = [[0, 1], [1, 0]]
        bp_bounding_boxes_input = {
            "systolic": [[0, 0, 1, 1], [1, 0, 2, 1]],
            "diastolic": [[0, 1, 1, 2], [1, 1, 2, 2]],
        }
        true_output = (dist_input, [])
        self.assertEqual(
            blood_pressure.filter_non_matches(dist_input, bp_bounding_boxes_input),
            true_output,
        )

        # Test the transposing feature.
        dist_input = [[0, 1]]
        bp_bounding_boxes_input = {
            "diastolic": [[0, 0, 1, 1], [1, 0, 2, 1]],
            "systolic": [[0, 1, 1, 2]],
        }
        print(blood_pressure.generate_x_dists_matrix(bp_bounding_boxes_input))
        print(blood_pressure.filter_non_matches(dist_input, bp_bounding_boxes_input))


if __name__ == "__main__":
    unittest.main()
