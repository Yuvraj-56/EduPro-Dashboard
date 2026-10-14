
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Excel file path
FILE_PATH = BASE_DIR / "data" / "EduPro Online Platform.xlsx"


def validate_data():
    # Load all required sheets
    users = pd.read_excel(FILE_PATH, sheet_name="Users")
    teachers = pd.read_excel(FILE_PATH, sheet_name="Teachers")
    courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
    transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")

    datasets = {
        "Users": users,
        "Teachers": teachers,
        "Courses": courses,
        "Transactions": transactions,
    }

    print("\n" + "=" * 60)
    print("EDUPRO DATA QUALITY VALIDATION")
    print("=" * 60)

    # 1. Basic dataset information
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(f"Missing values: {df.isna().sum().sum()}")

        missing = df.isna().sum()
        if missing.any():
            print("Columns with missing values:")
            print(missing[missing > 0])

    # 2. Check unique identifiers
    print("\n--- UNIQUE ID CHECKS ---")

    id_columns = {
        "Users": (users, "UserID"),
        "Teachers": (teachers, "TeacherID"),
        "Courses": (courses, "CourseID"),
        "Transactions": (transactions, "TransactionID"),
    }

    for name, (df, column) in id_columns.items():
        print(
            f"{name} duplicate {column}: "
            f"{df[column].duplicated().sum()}"
        )

    # 3. Check transaction references
    print("\n--- RELATIONSHIP CHECKS ---")

    invalid_users = ~transactions["UserID"].isin(users["UserID"])
    invalid_courses = ~transactions["CourseID"].isin(courses["CourseID"])
    invalid_teachers = ~transactions["TeacherID"].isin(teachers["TeacherID"])

    print(f"Transactions with unknown users: {invalid_users.sum()}")
    print(f"Transactions with unknown courses: {invalid_courses.sum()}")
    print(f"Transactions with unknown teachers: {invalid_teachers.sum()}")

    # 4. Check numeric values
    print("\n--- NUMERIC VALIDATION ---")

    print("Negative transaction amounts:", (transactions["Amount"] < 0).sum())
    print("Negative course prices:", (courses["CoursePrice"] < 0).sum())
    print("Invalid course ratings:",
          ((courses["CourseRating"] < 0) |
           (courses["CourseRating"] > 5)).sum())

    print("Invalid teacher ratings:",
          ((teachers["TeacherRating"] < 0) |
           (teachers["TeacherRating"] > 5)).sum())

    # 5. Check transaction dates
    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"], errors="coerce"
    )

    print("Invalid transaction dates:",
          transactions["TransactionDate"].isna().sum())

    print("\nValidation completed.")


if __name__ == "__main__":
    validate_data()
