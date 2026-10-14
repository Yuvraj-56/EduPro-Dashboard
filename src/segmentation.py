"""Learner segmentation and personalized course recommendation logic."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import build_learner_profiles


NUMERIC_FEATURES = [
    "TotalEnrollments",
    "UniqueCourses",
    "TotalSpent",
    "AveragePayment",
    "AverageCourseRating",
    "AverageCoursesPerCategory",
    "EnrollmentFrequency",
    "DiversityScore",
    "LearningDepthIndex",
]
CATEGORICAL_FEATURES = [
    "PreferredCategory",
    "PreferredCourseType",
    "PreferredLevel",
]
ARCHETYPE_NAMES = [
    "Curious Explorers",
    "Career Accelerators",
    "Focused Specialists",
]


@dataclass
class SegmentationResult:
    profiles: pd.DataFrame
    feature_matrix: np.ndarray
    candidate_metrics: pd.DataFrame
    selected_k: int
    silhouette: float
    hierarchical_silhouette: float


def load_workbook_data(workbook_path):
    """Load the three source sheets used by the learner analytics."""
    users = pd.read_excel(workbook_path, sheet_name="Users")
    courses = pd.read_excel(workbook_path, sheet_name="Courses")
    transactions = pd.read_excel(workbook_path, sheet_name="Transactions")
    return users, courses, transactions


def _feature_matrix(profiles):
    numeric_profiles = profiles[NUMERIC_FEATURES].copy()
    sparse_features = [
        "TotalEnrollments",
        "UniqueCourses",
        "TotalSpent",
        "AveragePayment",
        "AverageCoursesPerCategory",
        "EnrollmentFrequency",
    ]
    numeric_profiles[sparse_features] = np.log1p(
        numeric_profiles[sparse_features].clip(lower=0)
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(range(len(NUMERIC_FEATURES)))),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(
                    range(
                        len(NUMERIC_FEATURES),
                        len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
                    )
                ),
            ),
        ],
        remainder="drop",
    )
    model_input = pd.concat(
        [
            numeric_profiles.reset_index(drop=True),
            profiles[CATEGORICAL_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    matrix = preprocessor.fit_transform(model_input)
    return np.asarray(matrix, dtype=float)


def _segment_names(profiles):
    stats = profiles.groupby("Cluster", observed=True).agg(
        categories=("CategoriesExplored", "mean"),
        enrollments=("TotalEnrollments", "mean"),
        spending=("AveragePayment", "mean"),
        depth=("LearningDepthIndex", "mean"),
    )
    normalized = stats.copy()
    for column in normalized:
        spread = stats[column].max() - stats[column].min()
        normalized[column] = (
            (stats[column] - stats[column].min()) / spread if spread else 0.5
        )

    scores = pd.DataFrame(index=stats.index)
    scores["Curious Explorers"] = (
        normalized["categories"] * 0.75
        + normalized["enrollments"] * 0.25
    )
    scores["Career Accelerators"] = (
        normalized["spending"] * 0.55
        + normalized["depth"] * 0.25
        + normalized["enrollments"] * 0.20
    )
    scores["Focused Specialists"] = (
        normalized["enrollments"] * 0.55
        + (1 - normalized["categories"]) * 0.30
        + normalized["depth"] * 0.15
    )

    labels = {}
    remaining = list(ARCHETYPE_NAMES)
    ranked = sorted(
        scores.index,
        key=lambda cluster: (
            scores.loc[cluster].max(),
            -cluster,
        ),
        reverse=True,
    )
    for index, cluster in enumerate(ranked):
        if remaining:
            archetype = scores.loc[cluster, remaining].idxmax()
            labels[cluster] = archetype
            remaining.remove(archetype)
        else:
            labels[cluster] = f"Learning Cohort {index + 1}"
    return labels


def _sample_indices(size, maximum=1000):
    if size <= maximum:
        return np.arange(size)
    return np.random.default_rng(42).choice(size, maximum, replace=False)


def _safe_silhouette(matrix, labels):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(matrix, labels))
    except ValueError:
        return float("nan")


def _hierarchical_validation(matrix, labels, sample_indices):
    if len(sample_indices) < 3 or len(np.unique(labels[sample_indices])) < 2:
        return float("nan")
    sampled_matrix = matrix[sample_indices]
    try:
        hierarchical = AgglomerativeClustering(
            n_clusters=len(np.unique(labels[sample_indices])),
            linkage="ward",
        )
        validation_labels = hierarchical.fit_predict(sampled_matrix)
        return _safe_silhouette(sampled_matrix, validation_labels)
    except ValueError:
        return float("nan")


def segment_learners(profiles, n_clusters=None):
    """Choose or apply K-Means clusters and validate with a hierarchy sample."""
    if profiles.empty:
        raise ValueError("Cannot segment an empty learner profile dataset.")
    if len(profiles) < 3:
        raise ValueError("At least three learner profiles are required.")

    profiles = profiles.reset_index(drop=True)
    matrix = _feature_matrix(profiles)
    candidates = range(2, min(8, len(profiles) - 1) + 1)
    sample_indices = _sample_indices(len(profiles))
    candidate_results = []
    fitted_models = {}

    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        score = _safe_silhouette(
            matrix[sample_indices], labels[sample_indices]
        )
        candidate_results.append(
            {"Clusters": k, "Silhouette": score, "Inertia": model.inertia_}
        )
        fitted_models[k] = (model, labels, score)

    candidate_metrics = pd.DataFrame(candidate_results)
    valid_scores = candidate_metrics.dropna(subset=["Silhouette"])
    if valid_scores.empty:
        raise ValueError("The available learner profiles cannot form clusters.")
    recommended_k = int(
        valid_scores.sort_values(
            ["Silhouette", "Clusters"], ascending=[False, True]
        ).iloc[0]["Clusters"]
    )

    selected_k = recommended_k if n_clusters is None else int(n_clusters)
    if selected_k < 2 or selected_k >= len(profiles):
        raise ValueError("Cluster count must be between 2 and learner count - 1.")
    if selected_k not in fitted_models:
        model = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        score = _safe_silhouette(
            matrix[sample_indices], labels[sample_indices]
        )
    else:
        model, labels, score = fitted_models[selected_k]

    result_profiles = profiles.copy()
    result_profiles["Cluster"] = labels
    segment_names = _segment_names(result_profiles)
    result_profiles["Segment"] = result_profiles["Cluster"].map(segment_names)

    if matrix.shape[1] >= 2:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(matrix)
        result_profiles["Dimension 1"] = coordinates[:, 0]
        result_profiles["Dimension 2"] = coordinates[:, 1]
    else:
        result_profiles["Dimension 1"] = matrix[:, 0]
        result_profiles["Dimension 2"] = 0.0

    consistency = {}
    for cluster, group in result_profiles.groupby("Cluster"):
        points = matrix[group.index.to_numpy()]
        centroid = points.mean(axis=0, keepdims=True)
        similarities = cosine_similarity(points, centroid).ravel()
        consistency[cluster] = float(np.clip(similarities.mean(), -1, 1))
    result_profiles["IntraClusterConsistency"] = result_profiles["Cluster"].map(
        consistency
    )
    hierarchical_score = _hierarchical_validation(
        matrix, labels, sample_indices
    )

    return SegmentationResult(
        profiles=result_profiles,
        feature_matrix=matrix,
        candidate_metrics=candidate_metrics,
        selected_k=selected_k,
        silhouette=score,
        hierarchical_silhouette=hierarchical_score,
    )


def recommend_courses(
    user_id,
    result,
    courses,
    transactions,
    category=None,
    level=None,
    course_type=None,
    limit=8,
):
    """Rank unowned courses using content, cluster, similar-learner, and rating signals."""
    profiles = result.profiles.reset_index(drop=True)
    target_matches = profiles.index[profiles["UserID"].eq(user_id)].tolist()
    if not target_matches:
        raise ValueError(f"Learner {user_id!r} was not found in the profiles.")
    target_index = target_matches[0]
    learner = profiles.iloc[target_index]

    enrolled = set(
        transactions.loc[
            transactions["UserID"].eq(user_id), "CourseID"
        ].dropna()
    )
    candidates = courses.loc[~courses["CourseID"].isin(enrolled)].copy()
    if category and category != "All categories":
        candidates = candidates.loc[candidates["CourseCategory"].eq(category)]
    if level and level != "All levels":
        candidates = candidates.loc[candidates["CourseLevel"].eq(level)]
    if course_type and course_type != "All course types":
        candidates = candidates.loc[candidates["CourseType"].eq(course_type)]
    if candidates.empty:
        return candidates.assign(
            RelevanceScore=pd.Series(dtype=float),
            RecommendationReason=pd.Series(dtype=str),
        )

    enrolled_courses = transactions.merge(
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
        how="inner",
    )
    enrolled_courses = enrolled_courses.merge(
        profiles[["UserID", "Cluster"]], on="UserID", how="inner"
    )

    cluster_counts = enrolled_courses.loc[
        enrolled_courses["Cluster"].eq(learner["Cluster"]), "CourseID"
    ].value_counts()
    cluster_maximum = cluster_counts.max() if not cluster_counts.empty else 1
    cluster_scores = cluster_counts / max(cluster_maximum, 1)

    distances = np.linalg.norm(
        result.feature_matrix - result.feature_matrix[target_index], axis=1
    )
    nearest = np.argsort(distances)
    nearest = [
        index
        for index in nearest
        if index != target_index
        and profiles.iloc[index]["UserID"] != user_id
    ][:30]
    peer_counts = {}
    for peer_index in nearest:
        peer_id = profiles.iloc[peer_index]["UserID"]
        weight = 1 / (0.1 + distances[peer_index])
        peer_courses = transactions.loc[
            transactions["UserID"].eq(peer_id), "CourseID"
        ]
        for course_id in peer_courses:
            peer_counts[course_id] = peer_counts.get(course_id, 0.0) + weight
    peer_maximum = max(peer_counts.values(), default=0.0)

    global_counts = enrolled_courses["CourseID"].value_counts()
    global_maximum = global_counts.max() if not global_counts.empty else 1

    candidates["ClusterPopularity"] = (
        candidates["CourseID"].map(cluster_scores).fillna(0)
    )
    candidates["SimilarLearnerSignal"] = (
        candidates["CourseID"].map(peer_counts).fillna(0) / peer_maximum
        if peer_maximum
        else 0.0
    )
    learner_history = enrolled_courses.loc[
        enrolled_courses["UserID"].eq(user_id)
    ]

    def preference_scores(column, candidate_column):
        preferences = learner_history[column].value_counts(normalize=True)
        return candidates[candidate_column].map(preferences).fillna(0)

    candidates["CategoryMatch"] = preference_scores(
        "CourseCategory", "CourseCategory"
    )
    candidates["LevelMatch"] = preference_scores("CourseLevel", "CourseLevel")
    candidates["CourseTypeMatch"] = preference_scores(
        "CourseType", "CourseType"
    )
    candidates["RatingSignal"] = (
        candidates["CourseRating"].fillna(0).clip(0, 5) / 5
    )
    candidates["PlatformPopularity"] = (
        candidates["CourseID"].map(global_counts).fillna(0) / global_maximum
    )
    candidates["RelevanceScore"] = 100 * (
        candidates["ClusterPopularity"] * 0.25
        + candidates["SimilarLearnerSignal"] * 0.20
        + candidates["CategoryMatch"] * 0.16
        + candidates["LevelMatch"] * 0.08
        + candidates["CourseTypeMatch"] * 0.08
        + candidates["RatingSignal"] * 0.15
        + candidates["PlatformPopularity"] * 0.08
    )

    def recommendation_reason(course):
        reasons = []
        if course["CourseCategory"] == learner["PreferredCategory"]:
            reasons.append("matches preferred category")
        if course["CourseLevel"] == learner["PreferredLevel"]:
            reasons.append("matches preferred level")
        if course["CourseType"] == learner["PreferredCourseType"]:
            reasons.append("matches preferred course type")
        if course["ClusterPopularity"] >= 0.5:
            reasons.append("popular with this segment")
        if course["SimilarLearnerSignal"] >= 0.5:
            reasons.append("chosen by similar learners")
        if not reasons:
            reasons.append("highly rated across the platform")
        return " · ".join(reasons[:3])

    candidates["RecommendationReason"] = candidates.apply(
        recommendation_reason, axis=1
    )
    return candidates.sort_values(
        ["RelevanceScore", "CourseRating"], ascending=[False, False]
    ).head(limit)
