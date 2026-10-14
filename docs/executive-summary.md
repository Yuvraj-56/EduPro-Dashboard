# EduPro Learner Intelligence — Executive Summary

## The opportunity

Learners have different course-taking patterns, but a one-size-fits-all
discovery experience does not reflect those differences. EduPro's current
workbook makes it possible to begin personalizing discovery from actual
enrollment behavior.

## What the data says

The supplied data covers **3,000 learners**, **60 courses**, and **10,000
enrollments** in 2025. All learners in the workbook have at least one
enrollment.

A behavior-based model selected **two cohorts** from the candidate cluster
counts:

- **Focused Specialists:** 2,540 learners (**84.7%**), averaging **1.53
  enrollments** and **1.48 course categories explored** per learner.
- **Curious Explorers:** 460 learners (**15.3%**), averaging **13.27
  enrollments** and **8.65 categories explored** per learner.

The 2-cluster solution has a sampled silhouette of **0.453**; a separate
hierarchical validation sample scores **0.452**. The smaller cohort's mean
enrollment count is about **298% above** the overall mean of 3.33. This is a
descriptive difference between groups, not proof that a program or
recommendation caused higher engagement.

## What EduPro can do now

The dashboard lets teams explore a learner profile, see its assigned
behavioral cohort, and filter a ranked shortlist by course category and
level or course type. It also assembles the highest-ranked courses into a
three-step relevance-ordered discovery path (not a prerequisite sequence).
The ranking combines content-based category, level,
and course-type preference fit; courses popular with the learner's cohort;
choices from similar learner profiles; course ratings; and platform
popularity. Courses already taken are excluded.

Age and gender are available for descriptive context and aggregate fairness
monitoring; neither is used to assign cohorts or rank recommendations. Names
and email addresses are not displayed.

## What cannot yet be claimed

The workbook does not record recommendation impressions, clicks, course
completion, retention, or satisfaction after recommendation. Therefore it
cannot measure recommendation precision, prove that recommendations improve
engagement, or estimate causal impact. Match scores are ranking signals, not
outcome probabilities.

## Recommended decision

Approve a **measured pilot**, not an unmonitored full-scale rollout:

1. Instrument recommendation exposure, rank, click, enrollment, and
   completion.
2. Compare the personalized shortlist with a popularity-only control.
3. Review recommendation quality and outcomes across learner subgroups.
4. Revisit the two-cohort split with educators; the current model mainly
   separates high enrollment breadth and volume from one-or-few enrollments.
5. Refresh the cohorts regularly and let learners adjust or dismiss
   recommendations.

The Streamlit dashboard is available locally with `streamlit run app.py`.
