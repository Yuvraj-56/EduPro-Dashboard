from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import build_learner_profiles
from src.segmentation import (
    load_workbook_data,
    recommend_courses,
    segment_learners,
)


BASE_DIR = Path(__file__).resolve().parent
WORKBOOK_PATH = BASE_DIR / "data" / "EduPro Online Platform.xlsx"
PALETTE = ["#8A3FFC", "#E94F9B", "#F4B942", "#5E63D8", "#D94B78", "#B76BD4"]
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "DM Sans, sans-serif", "color": "#38264A"},
    "margin": {"l": 20, "r": 20, "t": 104, "b": 28},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.08,
        "x": 0.5,
        "xanchor": "center",
    },
}


st.set_page_config(
    page_title="EduPro | Learner Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def load_source_data(workbook_path, modified_at):
    return load_workbook_data(workbook_path)


@st.cache_data(show_spinner=False)
def make_profiles(users, courses, transactions):
    return build_learner_profiles(users, courses, transactions)


@st.cache_data(show_spinner=False)
def make_segments(profiles, cluster_count):
    return segment_learners(profiles, cluster_count)


def format_money(value):
    return f"${value:,.0f}"


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-orb"></div>
            <div class="hero-copy">
                <div class="eyebrow">{eyebrow}</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="hero-mark">E<span>.</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, accent="violet", animation_delay=0):
    st.markdown(
        f"""
        <div class="metric-card {accent}" style="--card-delay:{animation_delay}ms">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(figure, height=350):
    layout = {
        **PLOT_LAYOUT,
        "legend": {
            **PLOT_LAYOUT["legend"],
            "font": {"family": "DM Sans, sans-serif", "size": 12, "color": "#352344"},
            "title": {
                "font": {"family": "DM Sans, sans-serif", "size": 12, "color": "#352344"}
            },
        },
    }
    figure.update_layout(
        **layout,
        height=height,
        title={
            "x": 0.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "font": {"family": "DM Sans, sans-serif", "size": 18, "color": "#352344"},
        },
        annotationdefaults={"font": {"family": "DM Sans, sans-serif", "size": 12, "color": "#493456"}},
    )
    figure.update_annotations(font={"family": "DM Sans, sans-serif", "size": 12, "color": "#493456"})
    figure.update_xaxes(
        showgrid=False,
        linecolor="#D9C7E2",
        tickfont={"family": "DM Sans, sans-serif", "size": 12, "color": "#493456"},
        title_font={"family": "DM Sans, sans-serif", "size": 13, "color": "#352344"},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#E9DDED",
        zeroline=False,
        linecolor="#D9C7E2",
        tickfont={"family": "DM Sans, sans-serif", "size": 12, "color": "#493456"},
        title_font={"family": "DM Sans, sans-serif", "size": 13, "color": "#352344"},
    )
    return figure


def segment_summary(result):
    profiles = result.profiles
    return (
        profiles.groupby("Segment", as_index=False)
        .agg(
            Learners=("UserID", "count"),
            Share=("UserID", lambda values: len(values) / len(profiles)),
            AvgEnrollments=("TotalEnrollments", "mean"),
            AvgCategories=("CategoriesExplored", "mean"),
            AvgCoursesPerCategory=("AverageCoursesPerCategory", "mean"),
            AvgFrequency=("EnrollmentFrequency", "mean"),
            AvgDiversity=("DiversityScore", "mean"),
            AvgSpend=("AveragePayment", "mean"),
            AvgTotalSpent=("TotalSpent", "mean"),
            AvgRating=("AverageCourseRating", "mean"),
            AvgDepth=("LearningDepthIndex", "mean"),
            Consistency=("IntraClusterConsistency", "mean"),
            MeanAge=("Age", "mean"),
            FemaleShare=("Gender", lambda values: values.eq("Female").mean()),
            PreferredCategory=("PreferredCategory", lambda values: values.mode().iloc[0]),
            PreferredLevel=("PreferredLevel", lambda values: values.mode().iloc[0]),
            PreferredCourseType=(
                "PreferredCourseType",
                lambda values: values.mode().iloc[0],
            ),
        )
        .sort_values("Learners", ascending=False)
    )


def overview_page(users, courses, transactions, result):
    profiles = result.profiles
    summary = segment_summary(result)
    hero(
        "EDUPRO LEARNER INTELLIGENCE",
        "Make every learning journey count.",
        "Understand learner behavior, spot meaningful cohorts, and turn course discovery into a more personal experience.",
    )

    total_learners = profiles["UserID"].nunique()
    average_rating = courses["CourseRating"].mean()
    active_share = (profiles["TotalEnrollments"] > 0).mean()
    metrics = st.columns(5)
    cards = [
        ("Learners understood", f"{total_learners:,}", "individual learner profiles", "violet"),
        ("Learning journeys", f"{len(transactions):,}", "enrollments in the source data", "teal"),
        ("Course catalog", f"{len(courses):,}", "courses across learning levels", "amber"),
        ("Catalog rating", f"{average_rating:.2f} / 5", "mean course rating", "coral"),
        ("Learner coverage", f"{active_share:.0%}", "learners with enrollment history", "blue"),
    ]
    for index, (column, card) in enumerate(zip(metrics, cards)):
        with column:
            metric_card(*card, animation_delay=index * 70)

    st.markdown('<div class="section-heading"><span>01</span> The learner landscape</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        monthly = transactions.copy()
        monthly["TransactionDate"] = pd.to_datetime(
            monthly["TransactionDate"], errors="coerce"
        )
        monthly["Month"] = monthly["TransactionDate"].dt.to_period("M").dt.to_timestamp()
        monthly = monthly.dropna(subset=["Month"]).groupby("Month").size().rename("Enrollments").reset_index()
        chart = px.area(
            monthly,
            x="Month",
            y="Enrollments",
            markers=True,
            color_discrete_sequence=[PALETTE[0]],
            title="Enrollment rhythm",
        )
        chart.update_traces(line={"width": 3}, fillcolor="rgba(103,87,232,0.13)")
        st.plotly_chart(style_chart(chart, 360), width="stretch")
    with right:
        segment_chart = px.pie(
            summary,
            names="Segment",
            values="Learners",
            hole=0.68,
            color_discrete_sequence=PALETTE,
            title="Who is learning with EduPro?",
        )
        segment_chart.update_traces(
            textposition="outside",
            textinfo="percent",
            marker={"line": {"color": "#FFFFFF", "width": 3}},
        )
        st.plotly_chart(style_chart(segment_chart, 360), width="stretch")

    st.markdown('<div class="section-heading"><span>02</span> Cohort signals</div>', unsafe_allow_html=True)
    table = summary.copy()
    table["Share"] = table["Share"].map(lambda value: f"{value:.1%}")
    table["AvgEnrollments"] = table["AvgEnrollments"].map(lambda value: f"{value:.1f}")
    table["AvgCategories"] = table["AvgCategories"].map(lambda value: f"{value:.1f}")
    table["AvgCoursesPerCategory"] = table["AvgCoursesPerCategory"].map(
        lambda value: f"{value:.1f}"
    )
    table["AvgFrequency"] = table["AvgFrequency"].map(lambda value: f"{value:.2f}")
    table["AvgDiversity"] = table["AvgDiversity"].map(lambda value: f"{value:.1f}")
    table["AvgSpend"] = table["AvgSpend"].map(format_money)
    table["AvgTotalSpent"] = table["AvgTotalSpent"].map(format_money)
    table["AvgRating"] = table["AvgRating"].map(lambda value: f"{value:.2f} / 5")
    table["AvgDepth"] = table["AvgDepth"].map(lambda value: f"{value:+.2f}")
    table["Consistency"] = table["Consistency"].map(lambda value: f"{value:.0%}")
    st.dataframe(
        table.rename(
            columns={
                "Learners": "Learners",
                "Share": "Learner share",
                "AvgEnrollments": "Enrollments / learner",
                "AvgCategories": "Categories explored",
                "AvgCoursesPerCategory": "Unique courses / category",
                "AvgFrequency": "Enrollments / active month",
                "AvgDiversity": "Diversity score · categories",
                "AvgSpend": "Avg. spending / learner / enrollment",
                "AvgTotalSpent": "Mean learner transaction total*",
                "AvgRating": "Avg. course rating",
                "AvgDepth": "Learning depth",
                "Consistency": "Behavior consistency",
                "PreferredCategory": "Preferred category",
                "PreferredLevel": "Preferred level",
                "PreferredCourseType": "Preferred course type",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Learning depth ranges from −1 (beginner-only) to +1 (advanced-only), "
        "calculated as (advanced − beginner) ÷ (advanced + beginner); intermediate "
        "enrollments are excluded from this ratio. Diversity score is the number of "
        "distinct categories explored. Average spending is the mean transaction amount "
        "per learner enrollment; total spending is summed within the transaction window. "
        "Neither measure is revenue or lifetime value. "
        "Age and gender are shown only for descriptive context; they are excluded from clustering and recommendation ranking."
    )

    with st.expander("Project scope · source fields, features, and method"):
        scope_left, scope_right = st.columns(2)
        with scope_left:
            st.markdown(
                """
                **Source fields used**

                - **Users:** UserID, Age, Gender
                - **Courses:** CourseID, CourseCategory, CourseType, CourseLevel, CourseRating
                - **Transactions:** UserID, CourseID, TransactionDate, Amount

                **Learner-level features**

                - **Engagement:** total courses enrolled, average unique courses per category, enrollment frequency
                - **Preference:** preferred category, preferred course level and type, average enrolled-course rating
                - **Behavior:** average spending per learner per enrollment, diversity score (number of categories explored), learning depth (beginner-versus-advanced ratio)
                """
            )
        with scope_right:
            st.markdown(
                """
                **Data-science workflow**

                1. Aggregate transactions to learner profiles; retain demographic fields for context.
                2. Reduce sparse-feature skew with `log1p`, standardize numeric features, and one-hot encode preferences.
                3. Compare K-Means cluster counts with elbow inertia and silhouette; validate a deterministic sample with hierarchical clustering.
                4. Rank unseen courses using content-based preference fit, similar learner profiles, course popularity within each segment, and course ratings.

                **Evaluation:** silhouette and intra-cluster consistency are available now. Recommendation precision and engagement lift require recommendation exposure and outcome data; the dashboard reports them as not yet measured.
                """
            )

    engaged = summary.sort_values("AvgEnrollments", ascending=False).iloc[0]
    baseline = profiles["TotalEnrollments"].mean()
    lift = (engaged["AvgEnrollments"] / baseline - 1) * 100 if baseline else 0
    st.markdown(
        f"""
        <div class="insight-banner">
            <div class="insight-icon">✦</div>
            <div>
                <div class="insight-label">A signal worth exploring</div>
                <div class="insight-copy"><strong>{engaged["Segment"]}</strong> averages
                {engaged["AvgEnrollments"]:.1f} enrollments per learner, a
                {lift:+.0f}% difference from the all-learner average. This is an
                observed cohort comparison—not a causal engagement lift.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def learner_explorer_page(profiles, courses, transactions, result):
    hero(
        "LEARNER PROFILE EXPLORER",
        "See the person behind the pattern.",
        "Choose a learner to understand their learning style and discover a short list of next best courses.",
    )
    ids = profiles["UserID"].astype(str).sort_values().tolist()
    selected_id = st.selectbox("Select a learner profile", ids, index=0)
    learner = profiles.loc[profiles["UserID"].astype(str).eq(selected_id)].iloc[0]

    facts = [
        ("Segment", learner["Segment"]),
        ("Enrolled courses", f'{int(learner["TotalEnrollments"])}'),
        ("Diversity score", f'{int(learner["DiversityScore"])} categories'),
        ("Preferred category", str(learner["PreferredCategory"])),
        ("Preferred level", str(learner["PreferredLevel"])),
        ("Preferred type", str(learner["PreferredCourseType"])),
    ]
    for row_start in range(0, len(facts), 3):
        cols = st.columns(3, gap="medium")
        for offset, (label, value) in enumerate(facts[row_start : row_start + 3]):
            with cols[offset]:
                metric_card(
                    label,
                    value,
                    "learner profile",
                    "teal" if label == "Segment" else "violet",
                    animation_delay=(row_start + offset) * 55,
                )

    st.markdown("#### Learning snapshot")
    detail_columns = st.columns(2, gap="large")
    with detail_columns[0]:
        st.write(f'**Learner ID:** {selected_id}')
        st.write(f'**Age:** {int(learner["Age"]) if pd.notna(learner["Age"]) else "Not available"}')
        st.write(f'**Gender:** {learner["Gender"]}')
        st.write(f'**Learning cadence:** {learner["EnrollmentFrequency"]:.1f} enrollments per active month')
    with detail_columns[1]:
        st.write(f'**Average course rating:** {learner["AverageCourseRating"]:.2f} / 5')
        st.write(
            f'**Learning depth index:** {learner["LearningDepthIndex"]:+.2f} '
            "(advanced − beginner) ÷ (advanced + beginner)"
        )
        st.write(f'**Average courses per category:** {learner["AverageCoursesPerCategory"]:.1f}')
        st.write(f'**Average spending per learner:** {format_money(learner["AveragePayment"])} per enrollment')
    if learner["TotalEnrollments"] == 0:
        st.info("This is a cold-start profile. Recommendations rely on segment and platform signals until enrollment history is available.")
    else:
        st.success(
            f'{learner["Segment"]} are most often associated with '
            f'{learner["PreferredCategory"]} courses at the '
            f'{learner["PreferredLevel"]} level.'
        )

    st.markdown("#### Build a next-step shortlist")
    filter_cols = st.columns(3)
    category = filter_cols[0].selectbox(
        "Category",
        ["All categories"] + sorted(courses["CourseCategory"].dropna().unique().tolist()),
    )
    level = filter_cols[1].selectbox(
        "Level",
        ["All levels"] + sorted(courses["CourseLevel"].dropna().unique().tolist()),
    )
    course_type = filter_cols[2].selectbox(
        "Course type",
        ["All course types"] + sorted(courses["CourseType"].dropna().unique().tolist()),
    )
    recommendations = recommend_courses(
        learner["UserID"],
        result,
        courses,
        transactions,
        category=category,
        level=level,
        course_type=course_type,
        limit=8,
    )
    if recommendations.empty:
        st.info("No unowned courses match these filters. Try a broader category or level.")
    else:
        st.markdown("##### Suggested 3-step discovery path · ordered by match, not prerequisite")
        path_columns = st.columns(min(3, len(recommendations)))
        path_steps = ["01 · Start here", "02 · Build momentum", "03 · Stretch further"]
        for index, (_, course) in enumerate(recommendations.head(3).iterrows()):
            with path_columns[index]:
                metric_card(
                    path_steps[index],
                    course["CourseName"],
                    f'{course["CourseCategory"]} · {course["CourseLevel"]} · {course["CourseType"]} · {course["RelevanceScore"]:.0f}% match',
                    ["violet", "teal", "amber"][index],
                    animation_delay=index * 90,
                )
        view = recommendations[
            [
                "CourseName",
                "CourseCategory",
                "CourseType",
                "CourseLevel",
                "CourseRating",
                "CoursePrice",
                "RelevanceScore",
                "RecommendationReason",
            ]
        ].rename(
            columns={
                "CourseName": "Course",
                "CourseCategory": "Category",
                "CourseType": "Type",
                "CourseLevel": "Level",
                "CourseRating": "Rating",
                "CoursePrice": "Price",
                "RelevanceScore": "Match",
                "RecommendationReason": "Why this fits",
            }
        )
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "Rating": st.column_config.NumberColumn(format="%.2f / 5"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Match": st.column_config.ProgressColumn(
                    help="Relative recommendation relevance score, not measured precision.",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                ),
            },
        )
        st.download_button(
            "Download shortlist",
            recommendations.to_csv(index=False).encode("utf-8"),
            file_name=f"edupro_recommendations_{selected_id}.csv",
            mime="text/csv",
        )
    st.caption(
        "Recommendations combine content-based category, level and course-type fit, "
        "similar-learner choices, popularity within the assigned segment, course rating, and platform popularity. "
        "The match score is a ranking signal—not measured recommendation precision."
    )


def segments_page(result):
    hero(
        "CLUSTERS & MODEL VALIDATION",
        "Find the shape of learner behavior.",
        "Compare unsupervised cohorts, inspect their behavioral footprint, and review the evidence behind the selected segmentation.",
    )
    profiles = result.profiles
    summary = segment_summary(result)
    cards = st.columns(4)
    metrics = [
        ("Selected segments", str(result.selected_k), "K-Means clusters"),
        ("Silhouette score", f"{result.silhouette:.3f}", "observed on a seeded sample"),
        (
            "Hierarchical check",
            f"{result.hierarchical_silhouette:.3f}"
            if pd.notna(result.hierarchical_silhouette)
            else "N/A",
            "Agglomerative silhouette · sample",
        ),
        (
            "Mean consistency",
            f'{summary["Consistency"].mean():.0%}',
            "cosine similarity to cohort centroid",
        ),
    ]
    for index, (column, metric) in enumerate(zip(cards, metrics)):
        with column:
            metric_card(*metric, accent="teal", animation_delay=index * 90)

    scatter = px.scatter(
        profiles,
        x="Dimension 1",
        y="Dimension 2",
        color="Segment",
        hover_data=["UserID", "TotalEnrollments", "PreferredCategory"],
        color_discrete_sequence=PALETTE,
        opacity=0.72,
        title="Learner cohorts · PCA projection of standardized features",
    )
    scatter.update_traces(marker={"size": 7})
    st.plotly_chart(style_chart(scatter, 470), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        elbow = px.line(
            result.candidate_metrics,
            x="Clusters",
            y="Inertia",
            markers=True,
            color_discrete_sequence=[PALETTE[0]],
            title="Elbow view · within-cluster inertia",
        )
        st.plotly_chart(style_chart(elbow, 340), width="stretch")
    with right:
        silhouette = px.line(
            result.candidate_metrics,
            x="Clusters",
            y="Silhouette",
            markers=True,
            color_discrete_sequence=[PALETTE[1]],
            title="Model selection · sampled silhouette",
        )
        silhouette.add_vline(
            x=result.selected_k,
            line_dash="dot",
            line_color="#EC6B77",
            annotation_text=f"selected k = {result.selected_k}",
        )
        st.plotly_chart(style_chart(silhouette, 340), width="stretch")

    comparison = summary.melt(
        id_vars="Segment",
        value_vars=[
            "AvgEnrollments",
            "AvgCategories",
            "AvgSpend",
            "AvgRating",
            "AvgDepth",
        ],
        var_name="Measure",
        value_name="Average",
    )
    comparison["Measure"] = comparison["Measure"].map(
        {
            "AvgEnrollments": "Enrollments / learner",
            "AvgCategories": "Categories explored",
            "AvgSpend": "Avg. spending / enrollment",
            "AvgRating": "Average course rating",
            "AvgDepth": "Learning depth index",
        }
    )
    bars = px.bar(
        comparison,
        x="Segment",
        y="Average",
        color="Segment",
        barmode="group",
        facet_col="Measure",
        facet_col_wrap=2,
        facet_col_spacing=0.12,
        facet_row_spacing=0.12,
        color_discrete_sequence=PALETTE,
        title="Segment comparison · observed averages by measure",
    )
    bars.update_yaxes(matches=None, title_text=None)
    bars.update_xaxes(
        title_text=None,
        showticklabels=True,
        tickangle=-12,
        tickfont={"family": "DM Sans, sans-serif", "size": 11, "color": "#493456"},
    )
    bars.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1],
            font={"family": "DM Sans, sans-serif", "size": 13, "color": "#352344"},
        )
    )
    comparison_chart = style_chart(bars, 760)
    comparison_chart.update_layout(
        margin={"l": 56, "r": 28, "t": 148, "b": 60},
        legend={
            **comparison_chart.layout.legend.to_plotly_json(),
            "title": {
                "text": "Learner segment",
                "font": {
                    "family": "DM Sans, sans-serif",
                    "size": 13,
                    "color": "#352344",
                    "weight": "bold",
                },
            },
            "font": {"family": "DM Sans, sans-serif", "size": 12, "color": "#352344"},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.12,
            "yanchor": "bottom",
        },
    )
    st.plotly_chart(comparison_chart, width="stretch")
    st.caption(
        "Each panel shows one measure. Bar colors identify the learner segment; "
        "compare values within a panel because the measures use different scales."
    )
    demographic_view = summary[
        ["Segment", "Learners", "MeanAge", "FemaleShare"]
    ].copy()
    demographic_view["MeanAge"] = demographic_view["MeanAge"].map(
        lambda value: f"{value:.1f}"
    )
    demographic_view["FemaleShare"] = demographic_view["FemaleShare"].map(
        lambda value: f"{value:.1%}"
    )
    st.markdown("#### Demographic context · for fairness monitoring only")
    st.dataframe(
        demographic_view.rename(
            columns={
                "MeanAge": "Mean age",
                "FemaleShare": "Female share",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Demographic values do not contribute to segment assignment or course ranking. "
        "Review subgroup recommendation quality before any operational rollout."
    )

    st.markdown("#### Evaluation & validation")
    most_active = summary.sort_values("AvgEnrollments", ascending=False).iloc[0]
    all_learner_average = profiles["TotalEnrollments"].mean()
    engagement_proxy = (
        (most_active["AvgEnrollments"] / all_learner_average - 1) * 100
        if all_learner_average
        else float("nan")
    )
    evaluation = pd.DataFrame(
        [
            {
                "Metric": "Silhouette score",
                "Purpose": "Cluster quality",
                "Current result": f"{result.silhouette:.3f}",
                "Interpretation": "Sampled separation/compactness; higher is better",
            },
            {
                "Metric": "Intra-cluster similarity",
                "Purpose": "Behavioral consistency",
                "Current result": f'{summary["Consistency"].mean():.1%}',
                "Interpretation": "Mean cosine similarity to assigned segment centroid",
            },
            {
                "Metric": "Recommendation precision",
                "Purpose": "Recommendation relevance",
                "Current result": "Not measured",
                "Interpretation": "Needs judged relevance or recommendation response data",
            },
            {
                "Metric": "Engagement lift (proxy)",
                "Purpose": "Impact estimate",
                "Current result": f"{engagement_proxy:+.1f}% cohort enrollment difference",
                "Interpretation": "Descriptive vs all-learner mean; not causal or measured lift",
            },
        ]
    )
    st.dataframe(evaluation, hide_index=True, width="stretch")

    with st.expander("What these validation scores mean"):
        st.markdown(
            """
            - **K-Means silhouette** measures how well-separated and compact the selected clusters are; higher is better, with 1 as the upper bound. It is calculated on a deterministic sample of up to 1,000 learners.
            - **Hierarchical silhouette** independently checks a deterministic sample using agglomerative clustering. It is a validation comparison, not a second label assignment for the dashboard.
            - **Intra-cluster consistency** is the average cosine similarity of each learner's encoded profile to their cohort centroid.
            - **Elbow inertia** is the within-cluster sum of squared distances. The silhouette score is used for automatic selection; the elbow chart is provided as a complementary diagnostic.

            These are descriptive, unsupervised measures. They do not establish that a segment causes higher engagement or course completion.
            """
        )


def insights_page(users, courses, transactions, result):
    hero(
        "INSIGHTS & EXECUTIVE BRIEF",
        "Turn analysis into action.",
        "A concise, evidence-led readout for learning teams and public-sector stakeholders.",
    )
    profiles = result.profiles
    summary = segment_summary(result)
    categories = (
        transactions.merge(
            courses[["CourseID", "CourseCategory"]], on="CourseID", how="left"
        )["CourseCategory"]
        .value_counts()
        .head(8)
        .rename_axis("Category")
        .reset_index(name="Enrollments")
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        category_chart = px.bar(
            categories.sort_values("Enrollments"),
            x="Enrollments",
            y="Category",
            orientation="h",
            color="Enrollments",
            color_continuous_scale=["#D7D2FF", PALETTE[0]],
            title="Most explored learning categories",
        )
        category_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(category_chart, 380), width="stretch")
    with right:
        level_mix = (
            transactions.merge(courses[["CourseID", "CourseLevel"]], on="CourseID")
            ["CourseLevel"]
            .value_counts()
            .rename_axis("Level")
            .reset_index(name="Enrollments")
        )
        level_chart = px.pie(
            level_mix,
            names="Level",
            values="Enrollments",
            hole=0.62,
            color_discrete_sequence=PALETTE,
            title="Enrollment mix by course level",
        )
        st.plotly_chart(style_chart(level_chart, 380), width="stretch")

    lead_segment = summary.sort_values("Learners", ascending=False).iloc[0]
    high_activity = summary.sort_values("AvgEnrollments", ascending=False).iloc[0]
    mean_enrollments = profiles["TotalEnrollments"].mean()
    high_activity_difference = (
        (high_activity["AvgEnrollments"] / mean_enrollments - 1) * 100
        if mean_enrollments
        else 0
    )
    top_category = categories.iloc[0]["Category"] if not categories.empty else "Not available"
    executive_summary = f"""EDUPRO | EXECUTIVE SUMMARY

EduPro's workbook contains {len(users):,} learners, {len(courses):,} courses, and {len(transactions):,} enrollments. The learner profiles cover {profiles["UserID"].nunique():,} unique learners. A {result.selected_k}-cohort K-Means model was selected by the highest sampled silhouette score ({result.silhouette:.3f}); an independent hierarchical sample check scored {result.hierarchical_silhouette:.3f}.

The largest observed cohort is {lead_segment["Segment"]} ({int(lead_segment["Learners"]):,} learners; {lead_segment["Share"]:.1%}). {high_activity["Segment"]} has the highest mean enrollment count ({high_activity["AvgEnrollments"]:.1f} per learner), a {high_activity_difference:+.0f}% descriptive difference from the overall learner average. {top_category} is the most enrolled course category.

Recommended next steps:
1. Pilot the ranked, cluster-aware course shortlists and collect impressions, clicks, enrollments, and completions.
2. Compare recommendation outcomes against a simple popularity baseline, with opt-out and fairness monitoring.
3. Refit and review the segments as new enrollment and outcome data accumulates.

Interpretation: enrollment-frequency differences and recommendation match scores are descriptive proxies. The current workbook contains no click, completion, retention, or recommendation-exposure outcomes, so it cannot substantiate causal engagement lift or measured recommendation precision. Transaction amounts are not treated as revenue or lifetime value.
"""
    st.markdown("#### Executive summary")
    st.text_area("Copy-ready stakeholder brief", executive_summary, height=300)
    st.download_button(
        "Download executive summary",
        executive_summary.encode("utf-8"),
        file_name="edupro_executive_summary.txt",
        mime="text/plain",
        width="stretch",
    )
    st.markdown("#### Research findings at a glance")
    st.markdown(
        f"""
        - **Data footprint:** {len(users):,} learner records, {len(courses):,} catalog courses, and {len(transactions):,} transaction records.
        - **Segmentation:** {result.selected_k} behavioral cohorts selected by a sampled silhouette comparison across candidate cluster counts; hierarchical validation is shown alongside it.
        - **Discovery opportunity:** {top_category} leads the enrollment mix. Cohort-specific category and level preferences inform recommendation ranking.
        - **Measurement gap:** no impression, click, completion, retention, or satisfaction-after-recommendation fields are present. Precision and engagement lift require an instrumented pilot before they can be reported as outcomes.
        """
    )
    st.markdown("#### Practical rollout")
    st.markdown(
        """
        1. Start with a small, monitored pilot that records recommendation exposure, clicks, enrollments, and completion.
        2. Evaluate precision@k and completion/engagement lift against a popularity-only control.
        3. Check cohort stability and recommendation quality across age and gender groups before expanding.
        4. Refresh segments periodically and let learners refine category or level preferences.
        """
    )


def data_quality_page(users, courses, transactions, profiles):
    hero(
        "DATA QUALITY & COVERAGE",
        "Trust the signal before acting.",
        "Review workbook coverage, referential integrity, and missingness before interpreting cohort insights.",
    )
    expected_sheets = {
        "Users": (users, "UserID"),
        "Courses": (courses, "CourseID"),
        "Transactions": (transactions, "TransactionID"),
        "Learner profiles": (profiles, "UserID"),
    }
    quality_rows = []
    for name, (frame, identifier) in expected_sheets.items():
        quality_rows.append(
            {
                "Dataset": name,
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "Duplicate rows": int(frame.duplicated().sum()),
                "Missing cells": int(frame.isna().sum().sum()),
                f"Duplicate {identifier}": (
                    int(frame[identifier].duplicated().sum())
                    if identifier in frame.columns
                    else "N/A"
                ),
            }
        )
    st.dataframe(pd.DataFrame(quality_rows), hide_index=True, width="stretch")

    invalid_users = int((~transactions["UserID"].isin(users["UserID"])).sum())
    invalid_courses = int((~transactions["CourseID"].isin(courses["CourseID"])).sum())
    invalid_amounts = int((transactions["Amount"] < 0).sum())
    dates = pd.to_datetime(transactions["TransactionDate"], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    checks = st.columns(4)
    quality_cards = [
            ("Unknown learner refs", invalid_users, "transactions without a user"),
            ("Unknown course refs", invalid_courses, "transactions without a course"),
            ("Negative amounts", invalid_amounts, "transaction amount checks"),
            ("Invalid dates", invalid_dates, "unparseable enrollment dates"),
    ]
    for index, (col, (label, value, note)) in enumerate(zip(checks, quality_cards)):
        with col:
            metric_card(
                label,
                f"{value:,}",
                note,
                "coral" if value else "teal",
                animation_delay=index * 90,
            )

    missingness = (
        pd.concat(
            {
                name: frame.isna().mean().mul(100)
                for name, (frame, _) in expected_sheets.items()
            },
            axis=1,
        )
        .fillna(0)
        .rename_axis("Field")
        .reset_index()
        .melt(id_vars="Field", var_name="Dataset", value_name="Missing %")
    )
    missingness = missingness.loc[missingness["Missing %"] > 0]
    if missingness.empty:
        st.success("No missing values were found in the displayed source and profile fields.")
    else:
        missing_chart = px.bar(
            missingness,
            x="Field",
            y="Missing %",
            color="Dataset",
            barmode="group",
            color_discrete_sequence=PALETTE,
            title="Missingness by field",
        )
        st.plotly_chart(style_chart(missing_chart, 350), width="stretch")
    st.caption(
        "The dashboard does not display usernames or email addresses. "
        "The provided workbook includes transaction amount and demographics; "
        "review access and data-retention controls before operational use."
    )


def main():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root {
          --ink:#352344; --muted:#695979; --violet:#8244e8; --pink:#e94f9b;
          --yellow:#ffd84d; --line:#eadcf0;
        }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        .stApp {
          background:
            radial-gradient(ellipse at 91% 7%,rgba(255,216,77,.42),transparent 34%),
            radial-gradient(ellipse at 8% 24%,rgba(233,79,155,.26),transparent 39%),
            radial-gradient(ellipse at 78% 72%,rgba(138,63,252,.20),transparent 37%),
            radial-gradient(ellipse at 15% 88%,rgba(255,133,174,.22),transparent 34%),
            linear-gradient(135deg,#fff0f8 0%,#f1e5ff 52%,#fff3d5 100%);
          color:var(--ink);
        }
        [data-testid="stHeader"] { background:rgba(255,245,250,.96); }
        [data-testid="stHeader"] button, [data-testid="stHeader"] button * {
          color:#4a1d56 !important;
        }
        [data-testid="stSidebar"] {
          background:linear-gradient(165deg,#351746 0%,#54205f 54%,#71316f 100%);
          border-right:1px solid rgba(255,255,255,.12);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
        .block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1440px; }
        [data-testid="stMain"], [data-testid="stMain"] p,
        [data-testid="stMain"] li, [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
          color:var(--ink);
        }
        [data-testid="stMain"] h1, [data-testid="stMain"] h2,
        [data-testid="stMain"] h3, [data-testid="stMain"] h4,
        [data-testid="stMain"] h5, [data-testid="stMain"] h6 {
          color:#38234c !important;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {
          color:#685779 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {
          color:#fff7ff !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] input {
          accent-color:var(--yellow);
        }
        [data-testid="stSidebar"] hr {
          border-color:rgba(255,255,255,.2);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
          border-radius:10px;
          padding:5px 8px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
          background:rgba(255,216,77,.16);
        }
        .hero { position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between;
          min-height:206px; padding:35px 42px; margin:0 0 26px; border-radius:24px;
          background:linear-gradient(112deg,#4a1d56 0%,#9236a8 48%,#e94f9b 100%); color:white;
          box-shadow:0 18px 42px rgba(111,42,126,.22); }
        .hero-copy { position:relative; z-index:1; max-width:760px; min-width:0; flex:1; }
        .hero .eyebrow { font-size:11px; font-weight:800; letter-spacing:2px; color:#ffe66b; margin-bottom:12px; }
        .hero h1 { color:#fff !important; font-family:'Manrope',sans-serif; font-size:clamp(27px,3vw,40px); line-height:1.12;
          font-weight:800; letter-spacing:-1.3px; margin:0 0 10px; }
        .hero p { color:#fff0fb; font-size:15px; max-width:690px; margin:0; line-height:1.65; overflow-wrap:anywhere; }
        .hero-mark { position:relative; z-index:1; font:800 94px 'Manrope',sans-serif; color:rgba(255,255,255,.95);
          letter-spacing:-9px; padding-right:8px; }
        .hero-mark span { color:var(--yellow); }
        .hero-orb { position:absolute; width:320px;height:320px;border-radius:50%;right:90px;top:-190px;
          background:rgba(255,255,255,.08); box-shadow:0 0 0 40px rgba(255,255,255,.035),0 0 0 82px rgba(255,255,255,.025); }
        .metric-card { position:relative; isolation:isolate; overflow:hidden;
          box-sizing:border-box; height:168px; padding:18px 19px 16px; margin:0 0 12px; border:1px solid #eadcf0;
          border-radius:17px; background:rgba(255,255,255,.9); box-shadow:0 8px 22px rgba(87,43,104,.08);
          animation:card-enter .62s cubic-bezier(.2,.75,.25,1) both;
          animation-delay:var(--card-delay,0ms);
          display:flex; flex-direction:column; align-items:flex-start;
          transition:transform .3s cubic-bezier(.2,.8,.2,1),box-shadow .3s ease,border-color .3s ease,background .3s ease; }
        .metric-card:after { content:''; position:absolute; z-index:-1; pointer-events:none;
          top:-35%; left:-90%; width:55%; height:170%;
          background:linear-gradient(105deg,transparent 12%,rgba(255,255,255,.78) 48%,transparent 84%);
          transform:skewX(-18deg); transition:left .65s cubic-bezier(.2,.65,.3,1); }
        .metric-card:hover { transform:translateY(-8px) scale(1.025); border-color:#f4b942;
          background:linear-gradient(135deg,#fff0b8 0%,#ffd3e7 48%,#e8d7ff 100%);
          box-shadow:0 18px 34px rgba(87,43,104,.19); }
        .metric-card:hover .metric-label, .metric-card:hover .metric-value,
        .metric-card:hover .metric-note { color:#352344 !important; }
        .metric-card:hover:after { left:145%; }
        .metric-card:before { content:''; display:block; height:4px; width:30px; border-radius:3px; margin-bottom:14px;
          background:#8244e8; transition:width .3s ease; }
        .metric-card:hover:before { width:52px; }
        @keyframes card-enter {
          from { opacity:0; translate:0 18px; scale:.97; }
          to { opacity:1; translate:0 0; scale:1; }
        }
        .metric-card.teal:before { background:#e94f9b; } .metric-card.amber:before { background:#f4b942; }
        .metric-card.coral:before { background:#df5793; } .metric-card.blue:before { background:#5e63d8; }
        .metric-label { color:#695779; font:700 11px 'Manrope',sans-serif;
          letter-spacing:.7px; text-transform:uppercase; line-height:1.25; }
        .metric-value { color:#352344; font:700 22px 'Manrope',sans-serif;
          letter-spacing:-.45px; line-height:1.2; margin:7px 0 4px; overflow-wrap:break-word; }
        .metric-note { color:#756585; font:500 11px 'Manrope',sans-serif; line-height:1.3; margin-top:auto; }
        .section-heading { display:flex; gap:12px; align-items:center; margin:28px 0 16px; color:#38234c;
          font:800 20px 'Manrope',sans-serif; letter-spacing:-.4px; }
        .section-heading span { color:#a33eae; font-size:12px; letter-spacing:1px; }
        .insight-banner { display:flex; gap:16px; align-items:center; margin:18px 0 5px; padding:18px 21px;
          background:linear-gradient(100deg,#fff0f7,#f3eaff); border:1px solid #ead3f0; border-radius:16px; }
        .insight-icon { display:grid; place-items:center; width:38px; height:38px; flex:0 0 38px; border-radius:12px;
          color:#3d1e4c; background:var(--yellow); font-size:20px; }
        .insight-label { color:#8d36a0; font-size:10px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
        .insight-copy { color:#594568; font-size:14px; line-height:1.6; margin-top:3px; }
        .insight-copy strong { color:#352344; }
        div[data-testid="stDataFrame"] { border:1px solid #eadcf0; border-radius:14px; overflow:hidden; background:#fff; }
        div[data-testid="stPlotlyChart"] { background:rgba(255,255,255,.92); border:1px solid #eadcf0; border-radius:16px; padding:8px 10px; box-shadow:0 8px 22px rgba(87,43,104,.06); }
        [data-testid="stExpander"] { background:rgba(255,255,255,.92); border:1px solid #eadcf0; border-radius:14px; }
        [data-testid="stExpander"] details > summary {
          color:#352344 !important; background:#fff7fc !important; border-radius:13px;
          transition:background .18s ease,color .18s ease;
        }
        [data-testid="stExpander"] details > summary *,
        [data-testid="stExpander"] details > summary [data-testid="stIconMaterial"] {
          color:#352344 !important;
        }
        [data-testid="stExpander"] details > summary:hover,
        [data-testid="stExpander"] details > summary:focus-visible,
        [data-testid="stExpander"] details[open] > summary {
          color:#352344 !important; background:#f4e8ff !important;
        }
        [data-testid="stExpander"] details > summary:hover *,
        [data-testid="stExpander"] details > summary:focus-visible *,
        [data-testid="stExpander"] details[open] > summary * {
          color:#352344 !important;
        }
        [data-testid="stAlert"] { border-radius:12px; }
        [data-testid="stMain"] [data-testid="stAlert"] p { color:#38234c !important; }
        [data-testid="stMain"] input, [data-testid="stMain"] textarea {
          color:#352344 !important; background:#fff !important; border-color:#ddc9e8 !important;
        }
        [data-testid="stMain"] [data-baseweb="select"] > div {
          color:#352344; background:#fff; border-color:#ddc9e8;
        }
        [data-testid="stMain"] [data-testid="stSelectbox"] [role="group"] {
          background:#fff !important; border:1px solid #ddc9e8 !important;
          border-radius:10px !important; box-shadow:0 2px 8px rgba(87,43,104,.05);
          transition:border-color .18s ease,box-shadow .18s ease,background .18s ease;
        }
        [data-testid="stMain"] [data-testid="stSelectbox"] [role="group"]:hover,
        [data-testid="stMain"] [data-testid="stSelectbox"] [role="group"]:focus-within {
          background:#fff !important; border-color:#a65ac4 !important;
          box-shadow:0 0 0 3px rgba(166,90,196,.14);
        }
        [data-testid="stMain"] [data-testid="stSelectbox"] input {
          color:#352344 !important; background:transparent !important;
          -webkit-text-fill-color:#352344 !important;
        }
        [data-testid="stMain"] [data-testid="stSelectbox"] button {
          color:#5a3472 !important; background:#f4eafa !important;
          border-left:1px solid #eadcf0 !important; border-radius:0 9px 9px 0 !important;
          transition:color .18s ease,background .18s ease;
        }
        [data-testid="stMain"] [data-testid="stSelectbox"] button:hover,
        [data-testid="stMain"] [data-testid="stSelectbox"] button:focus-visible {
          color:#4a1d56 !important; background:#ffe6f2 !important;
        }
        [role="listbox"], [role="option"], [data-testid="stPopover"] {
          color:#352344 !important; background:#fff !important;
        }
        [role="option"]:hover, [role="option"][data-focused="true"],
        [role="option"][aria-selected="true"] {
          color:#352344 !important; background:#f5e8fb !important;
        }
        h3, h4 { font-family:'Manrope',sans-serif !important; color:#38234c !important; }
        .stButton button, .stDownloadButton button {
          border-radius:10px; font-weight:800; color:#fff; border:0;
          background:linear-gradient(110deg,#8244e8,#e94f9b);
        }
        .stButton button:hover, .stDownloadButton button:hover {
          color:#352344; background:linear-gradient(110deg,#ffd84d,#ffb7da); border:0;
        }
        [data-testid="stCaptionContainer"] { color:#685779; }
        @media (prefers-reduced-motion: reduce) {
          *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important;
            transition-duration:.01ms !important; scroll-behavior:auto !important; }
        }
        @media (max-width: 800px) { .hero { padding:28px 24px; min-height:180px; } .hero-mark { display:none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:11px;margin:0 0 5px;">
          <div style="display:grid;place-items:center;width:38px;height:38px;border-radius:12px;
            background:linear-gradient(135deg,#ffd84d,#ffb7da);color:#4a1d56;font:800 19px Manrope;">E</div>
          <div><div style="font:800 17px Manrope;color:#fff7ff">EduPro</div>
          <div style="font-size:10px;letter-spacing:1px;color:#f4dff2">LEARNER INTELLIGENCE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "WORKSPACE",
        [
            "Overview",
            "Learner explorer",
            "Segments & model",
            "Insights & brief",
            "Data quality",
        ],
        label_visibility="visible",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("PERSONALIZATION ENGINE")
    st.sidebar.markdown("**Learner-first. Evidence-led.**")
    st.sidebar.caption(
        "Recommendations are ranked from behavioral and catalog signals. "
        "No learner emails are exposed in this dashboard."
    )

    if not WORKBOOK_PATH.exists():
        st.error(f"Data workbook not found: {WORKBOOK_PATH}")
        st.stop()

    modified_at = WORKBOOK_PATH.stat().st_mtime
    with st.spinner("Reading learner journeys and building behavioral profiles…"):
        users, courses, transactions = load_source_data(
            str(WORKBOOK_PATH), modified_at
        )
        profiles = make_profiles(users, courses, transactions)

    choices = ["Auto (recommended)"] + list(
        range(2, min(8, len(profiles) - 1) + 1)
    )
    cluster_choice = st.sidebar.selectbox("COHORT COUNT", choices, index=0)
    cluster_count = None if cluster_choice == "Auto (recommended)" else cluster_choice
    with st.spinner("Finding learner cohorts and validating the segmentation…"):
        result = make_segments(profiles, cluster_count)

    st.sidebar.caption(
        f"{len(profiles):,} profiles · {result.selected_k} cohorts · "
        f"silhouette {result.silhouette:.3f}"
    )

    pages = {
        "Overview": lambda: overview_page(users, courses, transactions, result),
        "Learner explorer": lambda: learner_explorer_page(
            result.profiles, courses, transactions, result
        ),
        "Segments & model": lambda: segments_page(result),
        "Insights & brief": lambda: insights_page(
            users, courses, transactions, result
        ),
        "Data quality": lambda: data_quality_page(
            users, courses, transactions, profiles
        ),
    }
    pages[page]()
    st.markdown(
        "<div style='padding-top:28px;color:#a0a9b9;font-size:11px;text-align:center'>"
        "EDUPRO LEARNER INTELLIGENCE · Behavioral patterns are descriptive and should be validated before operational decisions."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
