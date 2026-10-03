from pathlib import Path
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "iranian_churn.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/563/data.csv"
TARGET_COLUMN = "Churn"
FEATURE_COLUMNS = [
    "Call Failure",
    "Complains",
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Age",
    "Customer Value",
]
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, "Status", TARGET_COLUMN]


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse inconsistent whitespace in dataset column names."""
    normalized = data.copy()
    normalized.columns = [" ".join(str(name).split()) for name in normalized.columns]
    return normalized


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Check the downloaded file has the expected UCI columns and target."""
    normalized = normalize_column_names(data)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(normalized.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns: " + ", ".join(missing_columns)
        )

    normalized = normalized[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=[TARGET_COLUMN])
    target_values = set(normalized[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("The Churn target must contain only 0 and 1 values.")
    if normalized.empty:
        raise ValueError("The dataset contains no usable rows.")
    return normalized


def download_dataset(force: bool = False) -> Path:
    """Download the real Iranian Churn dataset from the UCI repository."""
    if DATA_FILE.exists() and not force:
        return DATA_FILE

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = DATA_FILE.with_suffix(".csv.part")
    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            temporary_file.write_bytes(response.read())
        validate_dataset(pd.read_csv(temporary_file, nrows=5))
        temporary_file.replace(DATA_FILE)
    finally:
        temporary_file.unlink(missing_ok=True)
    return DATA_FILE


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read and validate the local dataset."""
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Download it with: python -m src.download_data"
        )
    return validate_dataset(pd.read_csv(path))