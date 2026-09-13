# Identifying High-Risk Patients for 30-Day Hospital Readmission Using Healthcare Data

A scalable, distributed machine learning framework built with **Apache Spark (PySpark)** to predict 30-day hospital readmission from clinical and demographic data. This project supports **Sustainable Development Goal 3 (SDG 3): Good Health and Well-Being** by helping healthcare providers identify high-risk patients for proactive intervention.

## Overview

Hospital readmission within 30 days increases healthcare costs and often signals gaps in patient care or discharge planning. This project analyzes 30,000 patient records to uncover the clinical and institutional factors most associated with readmission, and builds classification models to predict it at scale using PySpark's distributed computing framework.

**Research question:** How effectively can machine learning techniques be used to predict 30-day hospital readmission using healthcare patient data?

## Dataset

- **Source:** [Hospital Readmission Prediction (Synthetic Dataset)](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset) — Kaggle
- **Size:** 30,000 patient records, 12 variables
- **Features:** `patient_id`, `age`, `gender`, `blood_pressure`, `cholesterol`, `bmi`, `diabetes`, `hypertension`, `medication_count`, `length_of_stay`, `discharge_destination`
- **Target:** `readmitted_30_days` (Yes/No)

## Key Findings (EDA)

- The dataset is **imbalanced** — most patients were not readmitted within 30 days.
- Numerical features show **low linear correlation** with one another (minimal multicollinearity).
- **Chronic conditions** are strong risk factors: patients with diabetes only (13.3%) and both diabetes + hypertension (13.1%) show the highest readmission rates, vs. 9.7% for patients with neither.
- **Discharge destination matters**: patients discharged to rehabilitation centers (17.5%) and nursing facilities (16.8%) have notably higher readmission rates than those discharged home (10.0%).

## Methodology

1. **Preprocessing** — data quality checks, parsing `blood_pressure` into `systolic_bp`, binary encoding of the target variable.
2. **Feature Engineering** — `StringIndexer` + `OneHotEncoder` for categorical variables, `VectorAssembler` to build the final feature vector.
3. **Modeling** — Logistic Regression, Decision Tree, and Random Forest (PySpark MLlib), tuned via `CrossValidator` + `ParamGridBuilder` (3-fold CV).
4. **Class Imbalance Handling** — a `classWeight` column (based on the majority/minority class ratio) was passed to each model's `weightCol` to reduce bias toward the majority ("Not Readmitted") class.
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, AUC-ROC, plus **class-specific Precision/Recall for the Readmitted class** and confusion matrices, since overall/weighted metrics can be misleading on imbalanced data.

## Results

> **Note:** The written report (`Final_phase_Big_data.pdf`) reflects the **initial** models, trained *before* class-imbalance handling was added. The table below reflects the **updated** notebook (`Final_Phase_Hospital_Readmission_updated.ipynb`), after class weighting was applied. See [Class Imbalance Handling](#class-imbalance-handling-update) below for details.

| Model | Accuracy | F1-Score | AUC | Recall (Readmitted) |
|---|---|---|---|---|
| Logistic Regression | 0.876 | 0.819 | 0.588 | 43% |
| Decision Tree | 0.875 | 0.818 | 0.471 | 42% |
| Random Forest | 0.876 | 0.819 | 0.573 | 33% |

Applying class weighting substantially improved the models' ability to correctly flag at-risk patients (Recall for the Readmitted class rose from near-zero to 33–43%). AUC remained moderate across models, consistent with the EDA finding that individual features have weak linear correlation with the outcome — suggesting the ceiling on discrimination lies in the available features rather than in class-imbalance handling. Random Forest feature importance identified cholesterol, BMI, discharge destination, and age as the most influential predictors.

### Class Imbalance Handling (Update)

The original report and initial models achieved high accuracy but a low AUC (Decision Tree even scored below 0.5), because the dataset's class imbalance caused the models to lean heavily toward predicting "Not Readmitted." The notebook was later updated to add a `classWeight` column (based on the majority/minority class ratio) passed to each model's `weightCol`, which meaningfully improved Recall on the minority (Readmitted) class — the metric that matters most for identifying at-risk patients. **This update is reflected in the notebook, not yet in the PDF report.**

## Tech Stack

- Apache Spark / PySpark (`pyspark.sql`, `pyspark.ml`)
- Pandas, Matplotlib, Seaborn (visualization)
- Jupyter Notebook

## Repository Structure

```
├── Final_Phase_Hospital_Readmission.ipynb   # Full analysis: EDA, preprocessing, modeling, evaluation
├── Final_phase_Big_data.pdf                 # Written report / paper
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install pyspark pandas matplotlib seaborn
   ```
2. Place the dataset (`hospital_readmissions_30k.csv`) in the project root.
3. Open and run `Final_Phase_Hospital_Readmission.ipynb` top to bottom (Jupyter or Google Colab).

## Future Work

- Incorporate additional features (prior admissions, medication history) to improve discriminative power.
- Explore non-linear/deep learning models and real-time streaming data for prediction.
