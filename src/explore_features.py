
from feature_engineering import create_learner_features


def explore_features():
    df = create_learner_features()

    print("\n" + "=" * 55)
    print("LEARNER FEATURE EXPLORATION")
    print("=" * 55)

    numeric_columns = [
        "Age",
        "TotalEnrollments",
        "UniqueCourses",
        "TotalSpent",
        "AveragePayment",
        "AverageCourseRating",
        "CategoriesExplored",
    ]

    print("\nNUMERIC FEATURE SUMMARY")
    print(df[numeric_columns].describe().round(2))

    print("\nPREFERRED COURSE CATEGORY COUNTS")
    print(df["PreferredCategory"].value_counts())

    print("\nPREFERRED COURSE LEVEL COUNTS")
    print(df["PreferredLevel"].value_counts())

    print("\nGENDER COUNTS")
    print(df["Gender"].value_counts())

    print("\nLEARNER ACTIVITY")
    print("Learners with 1 enrollment:",
          (df["TotalEnrollments"] == 1).sum())
    print("Learners with 5+ enrollments:",
          (df["TotalEnrollments"] >= 5).sum())
    print("Learners with zero spending:",
          (df["TotalSpent"] == 0).sum())

    print("\nExploration completed.")


if __name__ == "__main__":
    explore_features()
