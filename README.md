# EduPro Learner Intelligence

An interactive Streamlit dashboard for learner segmentation and personalized
course discovery. It profiles enrollment behavior, finds learner cohorts with
K-Means, compares the selected solution with a hierarchical validation sample,
and ranks courses using content-based, cohort, similar-learner, preference,
and rating signals.

## Run locally

The project expects the source workbook at
`data/EduPro Online Platform.xlsx`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard offers Overview, Learner explorer, Segments & model, Insights &
brief, and Data quality views. Choose an automatic cluster count (selected by
sampled silhouette score) or set a count in the sidebar.

## Research paper

The project research paper is available as a formatted Word document at
`docs/edupro_research_paper.docx`. Its Markdown source is maintained at
`docs/research-paper.md`.

## Measurement notes

Recommendation match scores are ranking signals, not measured precision.
Enrollment differences between cohorts are descriptive, not causal engagement
lift. The workbook does not include impression, click, completion, retention,
or post-recommendation satisfaction outcomes, so a recommendation pilot should
instrument these before reporting impact.

The dashboard deliberately does not display learner usernames or email
addresses. Age and gender are available as descriptive learner context only;
they are excluded from clustering and recommendation ranking.

## Tests

```powershell
python -m unittest discover -s tests -v
```
