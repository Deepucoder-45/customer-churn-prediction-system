import argparse

from src.data_utils import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI churn dataset.")
    parser.add_argument(
        "--force", action="store_true", help="Download the file again if it exists."
    )
    arguments = parser.parse_args()
    dataset_path = download_dataset(force=arguments.force)
    print(f"Dataset is ready at: {dataset_path}")


if __name__ == "__main__":
    main()