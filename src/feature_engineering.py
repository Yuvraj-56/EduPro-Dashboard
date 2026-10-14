
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "EduPro Online Platform.xlsx"


def build_learner_profiles(users, courses, transactions):
    """Aggregate demographics, enrollment behavior, and course preferences."""
    transaction_data = transactions.merge(
        courses[
            [
                "CourseID",
                "CourseCategory",
                "CourseType",
                "CourseLevel",
                "CourseRating",
            ]
        ],
        on="CourseID",
        how="left",
    )
    transaction_data["TransactionDate"] = pd.to_datetime(
        transaction_data["TransactionDate"], errors="coerce"
    )

    def preferred_value(values):
        values = values.dropna()
        modes = values.mode()
        return modes.iloc[0] if not modes.empty else "No enrollment history"

    columns = [
        "UserID",
        "TotalEnrollments",
        "UniqueCourses",
        "TotalSpent",
        "AveragePayment",
        "PreferredCategory",
        "PreferredCourseType",
        "PreferredLevel",
        "AverageCourseRating",
        "CategoriesExplored",
        "AverageCoursesPerCategory",
        "ActiveMonths",
        "EnrollmentFrequency",
        "DiversityScore",
        "BeginnerEnrollments",
        "IntermediateEnrollments",
        "AdvancedEnrollments",
        "LearningDepthIndex",
    ]

    if transaction_data.empty:
        activity = pd.DataFrame(columns=columns)
    else:
        transaction_data["CourseLevelNormalized"] = (
            transaction_data["CourseLevel"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )
        transaction_data["TransactionMonth"] = (
            transaction_data["TransactionDate"].dt.to_period("M")
        )

        activity = transaction_data.groupby("UserID").agg(
            TotalEnrollments=("TransactionID", "count"),
            UniqueCourses=("CourseID", "nunique"),
            TotalSpent=("Amount", "sum"),
            AveragePayment=("Amount", "mean"),
            PreferredCategory=("CourseCategory", preferred_value),
            PreferredCourseType=("CourseType", preferred_value),
            PreferredLevel=("CourseLevel", preferred_value),
            AverageCourseRating=("CourseRating", "mean"),
            CategoriesExplored=("CourseCategory", "nunique"),
            ActiveMonths=("TransactionMonth", "nunique"),
            BeginnerEnrollments=(
                "CourseLevelNormalized",
                lambda values: values.eq("beginner").sum(),
            ),
            IntermediateEnrollments=(
                "CourseLevelNormalized",
                lambda values: values.eq("intermediate").sum(),
            ),
            AdvancedEnrollments=(
                "CourseLevelNormalized",
                lambda values: values.eq("advanced").sum(),
            ),
        ).reset_index()
        activity["EnrollmentFrequency"] = (
            activity["TotalEnrollments"]
            / activity["ActiveMonths"].replace(0, pd.NA)
        ).fillna(0)
        activity["DiversityScore"] = (
            activity["CategoriesExplored"]
        )
        activity["AverageCoursesPerCategory"] = (
            activity["UniqueCourses"]
            / activity["CategoriesExplored"].replace(0, pd.NA)
        ).fillna(0)
        depth_denominator = (
            activity["AdvancedEnrollments"] + activity["BeginnerEnrollments"]
        )
        activity["LearningDepthIndex"] = (
            activity["AdvancedEnrollments"]
            - activity["BeginnerEnrollments"]
        ) / depth_denominator.replace(0, pd.NA)
        activity["LearningDepthIndex"] = activity[
            "LearningDepthIndex"
        ].fillna(0)

    learner_features = users[["UserID", "Age", "Gender"]].merge(
        activity, on="UserID", how="left"
    )
    fill_zero = [
        "TotalEnrollments",
        "UniqueCourses",
        "TotalSpent",
        "AveragePayment",
        "AverageCourseRating",
        "CategoriesExplored",
        "AverageCoursesPerCategory",
        "ActiveMonths",
        "EnrollmentFrequency",
        "DiversityScore",
        "BeginnerEnrollments",
        "IntermediateEnrollments",
        "AdvancedEnrollments",
        "LearningDepthIndex",
    ]
    learner_features[fill_zero] = learner_features[fill_zero].fillna(0)
    learner_features["PreferredCategory"] = learner_features[
        "PreferredCategory"
    ].fillna("No enrollment history")
    learner_features["PreferredLevel"] = learner_features[
        "PreferredLevel"
    ].fillna("No enrollment history")
    learner_features["PreferredCourseType"] = learner_features[
        "PreferredCourseType"
    ].fillna("No enrollment history")

    return learner_features


def create_learner_features():
    """Load workbook data and return one feature row per learner."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Excel file not found at: {FILE_PATH}")

    users = pd.read_excel(FILE_PATH, sheet_name="Users")
    courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
    transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")
    return build_learner_profiles(users, courses, transactions)


if __name__ == "__main__":
    profiles = create_learner_features()
    print(f"Created {len(profiles):,} learner profiles.")
    print(f"Feature columns: {', '.join(profiles.columns)}")
    print(profiles.head().to_string(index=False))
