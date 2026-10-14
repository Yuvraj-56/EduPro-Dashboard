# Student Segmentation and Personalized Course Recommendation at EduPro

## Abstract

This study turns EduPro's learner, course, and transaction workbook into
behavioral learner profiles, exploratory segments, and ranked course
recommendations. Learner features summarize enrollment volume and cadence,
category breadth, preferred course category, type and level, ratings,
payments, and course-level mix. Standardized behavioral features are clustered using
K-Means; candidate cluster counts are compared using sampled silhouette
scores, and a sampled agglomerative solution provides an independent
hierarchical validation. Recommendations combine cohort popularity, similar
learner choices, learner preferences, and course ratings.

For the supplied workbook, the 2-cluster solution has a sampled silhouette of
**0.453**, with a hierarchical validation silhouette of **0.4515**. The
largest cohort contains **84.7%** of learners and averages **1.53 enrollments
per learner**; the smaller cohort contains **15.3%** and averages **13.27**.
These results primarily distinguish enrollment intensity. The workbook has no
recommendation exposure, click, course completion, retention, or post-course
satisfaction outcomes, so recommendation precision and causal engagement
lift cannot be measured from this data.

## 1. Context and objective

EduPro's generic course discovery does not account for the different ways
learners engage with its catalog. The objective of this analysis is to
understand learner behavior and make relevant next-course options easier to
discover—not to predict aggregate course demand.

The dashboard translates this objective into four capabilities:

1. Build one behavior profile per learner.
2. Find and compare behavior-led cohorts.
3. Rank courses a learner has not already taken.
4. Show what is measured, what is only a proxy, and what additional outcomes
   are needed for impact evaluation.

## 2. Data and scope

The input is `data/EduPro Online Platform.xlsx`, with four workbook sheets.
The core analysis uses:

| Sheet | Records | Role |
|---|---:|---|
| Users | 3,000 | Learner ID and descriptive age/gender context |
| Courses | 60 | Category, type, level, price, duration, and rating |
| Transactions | 10,000 | Learner/course enrollments, dates, and amounts |

All 3,000 users have at least one transaction in this workbook. The transaction
dates span **2025-01-01 through 2025-12-30**. The course catalog contains 12
categories. Teacher records are not used in segmentation or recommendation:
teacher assignment is not a learner preference or outcome in the stated
scope.

The supplied course mix is **35.7% Beginner**, **29.5% Intermediate**, and
**34.8% Advanced** enrollments. The most enrolled categories are Data Science
(916), Finance (864), and Web Development (844).

## 3. Learner feature engineering

Transaction rows are joined to course metadata and aggregated to one row per
`UserID`, then left-joined to the user table so learners with no enrollments
remain representable.

| Feature family | Derived features |
|---|---|
| Engagement | Total enrollments, unique courses, average courses per explored category, active enrollment months, and enrollments per active month |
| Preference | Most frequently enrolled category, course type, and level; average rating of enrolled courses |
| Behavioral | Average transaction amount, total transaction amount, categories explored, and diversity score |
| Learning depth | Beginner, Intermediate, and Advanced enrollment counts; depth index |

The **diversity score** is the number of distinct categories explored.
**Average courses per category** is the number
of unique courses enrolled in divided by the number of categories explored.
The **learning depth index** is
`(Advanced enrollments − Beginner enrollments) / (Advanced enrollments + Beginner enrollments)`,
bounded from −1 (beginner-only) to +1 (advanced-only). Intermediate
enrollments are excluded from this beginner-versus-advanced ratio. The index
is zero when no beginner or advanced enrollments are present. **Enrollment
frequency** is total enrollments divided by the number of distinct calendar
months with an enrollment, not the full elapsed calendar span.

Missing activity is represented with zero-valued numeric behavior features
and a `No enrollment history` preference label. This preserves support for
cold-start profiles in the recommendation logic.

Age and gender are retained for learner context and aggregate fairness
monitoring, but **are excluded from cluster assignment and recommendation
ranking**. No names or email addresses are displayed in the dashboard.

## 4. Segmentation methodology

1. Apply `log1p` to sparse, right-skewed enrollment and spending measures,
   then standardize the numeric engagement, preference, and behavioral features.
2. One-hot encode preferred category, course type, and course level.
3. Fit K-Means with a fixed random seed and multiple initializations.
4. Evaluate candidate cluster counts from 2 through 8 (limited by the number
   of profiles).
5. Select the count with the highest silhouette score on a deterministic
   sample of up to 1,000 learners. The dashboard also shows within-cluster
   inertia as an elbow diagnostic.
6. Independently fit agglomerative Ward clustering to the same deterministic
   validation sample and report its silhouette as a validation comparison.
7. Use a two-dimensional PCA projection only for visualization; it does not
   determine the clusters.

The behavioral cohort labels (for example, Curious Explorers and Focused
Specialists) are descriptive names assigned after clustering. They are not
input labels, supervised predictions, or claims about a learner's
motivation.

### Model-selection results

| Candidate clusters | Sampled silhouette | K-Means inertia |
|---:|---:|---:|
| 2 | **0.4530** | 20,059.25 |
| 3 | 0.3192 | 14,863.53 |
| 4 | 0.2881 | 12,920.17 |
| 5 | 0.2966 | 11,577.21 |
| 6 | 0.2836 | 10,719.26 |
| 7 | 0.2809 | 10,121.25 |
| 8 | 0.2791 | 9,574.89 |

The independent hierarchical silhouette is **0.4515**. The close score is
evidence that a two-way separation is reproducible under the selected
validation method; it is not evidence that a two-cohort product taxonomy is
the only useful segmentation.

## 5. Findings

| Observed cohort | Learners | Share | Mean enrollments | Median enrollments | Mean categories explored | Mean transaction amount per enrollment* |
|---|---:|---:|---:|---:|---:|---:|
| Focused Specialists | 2,540 | 84.7% | 1.53 | 1 | 1.48 | 88.56 |
| Curious Explorers | 460 | 15.3% | 13.27 | 13 | 8.65 | 93.69 |

\*`AveragePayment` is the mean `Amount` over that learner's transactions.
Amounts are shown in workbook units; they are not assumed to be revenue,
profit, or learner lifetime value.

Key interpretations:

- The largest separation is **enrollment intensity and breadth**. The smaller
  cohort has approximately 8.7 times as many enrollments and explores
  approximately 5.9 times as many categories on average.
- The higher-activity cohort's mean enrollment count is **about 298% above**
  the overall mean of 3.33. This is an observed cohort comparison, **not a
  causal engagement lift**.
- Mean enrolled-course ratings are similar (3.11 versus 3.14), and mean
  learning-depth indices are close to neutral (−0.04 versus +0.01). The
  supplied clustering therefore does not establish a strong quality or
  beginner-to-advanced distinction.
- The majority and smaller cohorts have similar mean age (24.98 and 24.95)
  and female share (50.7% and 50.2%). Demographics are not used by the model;
  aggregate comparison is included for monitoring, not as evidence of
  fairness across every subgroup.
- Course-category activity is comparatively distributed: Data Science leads
  with 916 enrollments, followed by Finance (864) and Web Development (844).
  Recommendation ranking can use each learner's own category preference
  rather than relying only on a platform-wide top list.

## 6. Personalized recommendation logic

The dashboard excludes every course already present in the learner's
transaction history and supports category, level, and course-type filters.
It ranks remaining courses using content-based preference distributions and
a weighted combination of:

- course popularity within the learner's assigned cluster;
- weighted course choices of up to 30 behaviorally similar learner profiles;
- preferred-category, preferred-level, and course-type matches;
- course rating; and
- platform-wide popularity.

The learner view also presents the top three ranked courses as a discovery
path. This is a relevance-ordered shortlist, not a prerequisite-aware
curriculum or a guaranteed learning sequence.

The displayed 0–100 **match score** is a normalized ranking score from these
signals. It is **not recommendation precision**, a probability of enrollment,
or evidence that the learner will complete a course. Category and level
filters are applied before the displayed shortlist is returned. Learners
without enrollment history can still receive recommendations from
platform/course quality signals and the profile similarity calculation.

## 7. Evaluation and limitations

| Measure | Current evidence | Interpretation |
|---|---|---|
| Cluster quality | K-Means sampled silhouette 0.4530 | Compactness/separation of the current feature representation |
| Hierarchical validation | Sampled silhouette 0.4515 | Independent sample clustering comparison |
| Intra-cluster consistency | Average cosine similarity to the cohort centroid | Behavioral similarity within a cohort |
| Recommendation precision | Not measurable from this workbook | Requires labeled relevance or user feedback |
| Engagement lift (proxy) | Highest-activity segment enrollment mean is about 298% above all-learner mean | Descriptive enrollment difference only; causal lift requires an instrumented comparison/control |

Important limitations:

1. The model learns from historical transactions. Prior exposure and course
   availability are not recorded, so observed enrollments are an incomplete
   measure of learner preference.
2. The selected two-cluster split is strongly shaped by the difference
   between one-or-few enrollments and high activity. The majority cohort is
   broad; smaller, actionable personas may require more longitudinal
   interaction data and a product-reviewed choice of cluster count.
3. Ratings are catalog ratings attached to enrolled courses, not
   learner-specific post-course satisfaction.
4. There are no course completion, assessment, click, recommendation
   impression, retention, or satisfaction-after-recommendation fields.
5. Transaction amounts should not be equated with revenue without a
   documented definition and accounting validation.
6. Segments are descriptive and may change as the workbook is refreshed.
   Learners should be able to refine preferences or ignore a recommendation.

## 8. Recommendations and next steps

1. **Pilot before scaling.** Log what was shown, the rank, learner response,
   enrollment, and completion.
2. **Establish a baseline.** Compare the current ranked list with a
   popularity-only control using precision@k, click-through, enrollment, and
   completion measures.
3. **Check subgroup outcomes.** Compare exposure, ranking, clicks, and
   completion across demographic groups before deployment; the current
   descriptive cohort table is not a fairness certification.
4. **Review the segment choice with educators.** The high silhouette supports
   the dominant activity split; test whether course teams need stable,
   interpretable sub-cohorts before increasing `k`.
5. **Refresh and monitor.** Recompute features on a regular schedule and
   record model and data versions so quality changes can be traced.

## 9. Reproducibility

Run the dashboard with `streamlit run app.py`. The feature aggregation and
segmentation functions are in `src/feature_engineering.py` and
`src/segmentation.py`. Focused regression checks run with:

```powershell
python -m unittest discover -s tests -v
```

All metrics in this report were computed from the workbook distributed with
the project and the deterministic clustering configuration described above.
