import unittest

import pandas as pd

from src.data_utils import FEATURE_COLUMNS, REQUIRED_COLUMNS, validate_dataset


class ValidateDatasetTests(unittest.TestCase):
    def make_sample(self) -> pd.DataFrame:
        values = {column: [1, 2] for column in REQUIRED_COLUMNS}
        values["Churn"] = [0, 1]
        values["Status"] = [1, 2]
        values["Subscription  Length"] = [12, 24]
        values["Call  Failure"] = [0, 3]
        return pd.DataFrame(values)

    def test_normalizes_source_column_spacing(self) -> None:
        cleaned = validate_dataset(self.make_sample())
        self.assertEqual(cleaned.columns.tolist(), REQUIRED_COLUMNS)
        self.assertEqual(cleaned["Subscription Length"].tolist(), [12, 24])

    def test_status_is_not_a_model_input(self) -> None:
        self.assertNotIn("Status", FEATURE_COLUMNS)

    def test_rejects_a_missing_required_column(self) -> None:
        sample = self.make_sample().drop(columns=["Customer Value"])
        with self.assertRaisesRegex(ValueError, "Customer Value"):
            validate_dataset(sample)

    def test_drops_rows_without_a_target(self) -> None:
        sample = self.make_sample()
        sample.loc[1, "Churn"] = None
        cleaned = validate_dataset(sample)
        self.assertEqual(len(cleaned), 1)


if __name__ == "__main__":
    unittest.main()