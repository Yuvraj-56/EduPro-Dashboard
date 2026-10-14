
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = BASE_DIR / "data" / "EduPro Online Platform.xlsx"


def load_data():
    """Load all sheets from the EduPro Excel workbook."""

    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f"Excel file not found at: {FILE_PATH}"
        )

    # Read the workbook sheet names
    excel_file = pd.ExcelFile(FILE_PATH)

    print("\nEDUPRO DATA LOADER")
    print("=" * 45)

    print("\nAvailable sheets:")
    for sheet in excel_file.sheet_names:
        print(f"- {sheet}")

    # Load each sheet into a DataFrame
    data = {
        sheet: pd.read_excel(excel_file, sheet_name=sheet)
        for sheet in excel_file.sheet_names
    }

    # Display a basic summary of each sheet
    for sheet_name, df in data.items():
        print(f"\n{'=' * 45}")
        print(f"SHEET: {sheet_name}")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nFirst 5 records:")
        print(df.head().to_string(index=False))

    return data


if __name__ == "__main__":
    load_data()
