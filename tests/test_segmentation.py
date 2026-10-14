import pandas as pd
import unittest

from src.feature_engineering import build_learner_profiles
from src.segmentation import _feature_matrix, recommend_courses, segment_learners


def sample_data():
    users = pd.DataFrame(
        {
            "UserID": [f"U{i}" for i in range(1, 7)],
            "Age": [20, 21, 33, 35, 45, 29],
            "Gender": ["Female", "Female", "Male", "Male", "Female", "Male"],
        }
    )
    courses = pd.DataFrame(
        {
            "CourseID": [f"C{i}" for i in range(1, 7)],
            "CourseName": [
                "Python Intro",
                "Python Advanced",
                "Design Intro",
                "Design Advanced",
                "Business Basics",
                "AI Foundations",
            ],
            "CourseCategory": [
                "Programming",
                "Programming",
                "Design",
                "Design",
                "Business",
                "AI",
            ],
            "CourseType": [
                "Paid",
                "Paid",
                "Free",
                "Paid",
                "Paid",
                "Free",
            ],
            "CourseLevel": [
                "Beginner",
                "Advanced",
                "Beginner",
                "Advanced",
                "Intermediate",
                "Intermediate",
            ],
            "CourseRating": [4.8, 4.6, 4.4, 4.7, 4.0, 4.9],
            "CoursePrice": [10, 20, 0, 15, 12, 18],
        }
    )
    transactions = pd.DataFrame(
        {
            "TransactionID": [f"T{i}" for i in range(1, 11)],
            "UserID": [
                "U1", "U1", "U2", "U2", "U3",
                "U3", "U4", "U4", "U5", "U6",
            ],
            "CourseID": ["C1", "C2", "C1", "C2", "C3", "C4", "C3", "C4", "C5", "C6"],
            "TransactionDate": pd.date_range("2025-01-01", periods=10, freq="30D"),
            "Amount": [10, 20, 10, 20, 0, 15, 0, 15, 12, 18],
        }
    )
    return users, courses, transactions


class SegmentationTests(unittest.TestCase):
    def test_profile_features_include_diversity_depth_and_cold_start(self):
        users, courses, transactions = sample_data()
        users.loc[len(users)] = ["U7", 27, "Female"]
        users.loc[len(users)] = ["U8", 31, "Male"]
        users.loc[len(users)] = ["U9", 38, "Female"]
        users.loc[len(users)] = ["U10", 26, "Male"]
        extra_transactions = pd.DataFrame(
            {
                "TransactionID": ["T11", "T12", "T13"],
                "UserID": ["U8", "U9", "U10"],
                "CourseID": ["C1", "C2", "C5"],
                "TransactionDate": pd.to_datetime(
                    ["2025-02-01", "2025-02-02", "2025-02-03"]
                ),
                "Amount": [10, 20, 12],
            }
        )
        transactions = pd.concat(
            [transactions, extra_transactions], ignore_index=True
        )
        profiles = build_learner_profiles(users, courses, transactions)

        beginner_heavy = profiles.set_index("UserID").loc["U1"]
        learner_profiles = profiles.set_index("UserID")
        cold_start = profiles.set_index("UserID").loc["U7"]

        self.assertEqual(beginner_heavy["TotalEnrollments"], 2)
        self.assertEqual(beginner_heavy["CategoriesExplored"], 1)
        self.assertEqual(beginner_heavy["AverageCoursesPerCategory"], 2)
        self.assertEqual(beginner_heavy["DiversityScore"], 1)
        self.assertEqual(beginner_heavy["LearningDepthIndex"], 0)
        self.assertEqual(learner_profiles.loc["U8", "LearningDepthIndex"], -1)
        self.assertEqual(learner_profiles.loc["U9", "LearningDepthIndex"], 1)
        self.assertEqual(learner_profiles.loc["U10", "LearningDepthIndex"], 0)
        self.assertEqual(beginner_heavy["PreferredCourseType"], "Paid")
        self.assertEqual(beginner_heavy["AveragePayment"], 15)
        self.assertEqual(cold_start["TotalEnrollments"], 0)
        self.assertEqual(cold_start["PreferredCategory"], "No enrollment history")

    def test_segmentation_and_recommendations_exclude_owned_courses(self):
        users, courses, transactions = sample_data()
        profiles = build_learner_profiles(users, courses, transactions)
        result = segment_learners(profiles, n_clusters=2)

        self.assertEqual(result.selected_k, 2)
        self.assertEqual(result.profiles["Segment"].nunique(), 2)
        self.assertTrue(result.candidate_metrics["Silhouette"].notna().any())
        self.assertTrue(
            result.profiles[["Dimension 1", "Dimension 2"]].notna().all().all()
        )

        recommendations = recommend_courses(
            "U1", result, courses, transactions, limit=6
        )
        self.assertFalse(set(recommendations["CourseID"]).intersection({"C1", "C2"}))
        self.assertTrue(recommendations["RelevanceScore"].between(0, 100).all())
        self.assertTrue(recommendations["RelevanceScore"].is_monotonic_decreasing)

    def test_recommendation_filters_return_only_matching_catalog_items(self):
        users, courses, transactions = sample_data()
        profiles = build_learner_profiles(users, courses, transactions)
        result = segment_learners(profiles, n_clusters=2)
        recommendations = recommend_courses(
            "U1",
            result,
            courses,
            transactions,
            category="Design",
            level="Advanced",
        )

        self.assertFalse(recommendations.empty)
        self.assertTrue(recommendations["CourseCategory"].eq("Design").all())
        self.assertTrue(recommendations["CourseLevel"].eq("Advanced").all())

    def test_course_type_filter_returns_only_matching_catalog_items(self):
        users, courses, transactions = sample_data()
        profiles = build_learner_profiles(users, courses, transactions)
        result = segment_learners(profiles, n_clusters=2)
        recommendations = recommend_courses(
            "U1",
            result,
            courses,
            transactions,
            course_type="Free",
        )

        self.assertFalse(recommendations.empty)
        self.assertTrue(recommendations["CourseType"].eq("Free").all())

    def test_demographics_do_not_change_clustering_features(self):
        users, courses, transactions = sample_data()
        profiles = build_learner_profiles(users, courses, transactions)
        changed_demographics = profiles.copy()
        changed_demographics["Age"] = changed_demographics["Age"] + 20
        changed_demographics["Gender"] = "Not specified"

        self.assertTrue(
            (
                _feature_matrix(profiles)
                == _feature_matrix(changed_demographics)
            ).all()
        )

    def test_cold_start_learner_gets_valid_recommendations(self):
        users, courses, transactions = sample_data()
        users.loc[len(users)] = ["U7", 27, "Female"]
        profiles = build_learner_profiles(users, courses, transactions)
        result = segment_learners(profiles, n_clusters=2)

        recommendations = recommend_courses(
            "U7", result, courses, transactions, limit=3
        )
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(recommendations["RelevanceScore"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
