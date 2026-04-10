# Machine Learning Assignment Plan

## Assignment Snapshot

- Course: Machine Learning
- Assignment: End-to-End Supervised Learning Pipeline
- Deliverables: 1 PDF report and 1 Jupyter notebook
- Submission window: March 15, 2026 to April 12, 2026, 11:59 PM IST
- Submission format: Zip all files, then upload through Lumen

## Core Requirements

We need to build a full supervised classification workflow on one dataset and cover all of the following:

1. Data quality checks and preprocessing
2. Baseline supervised models
3. Decision tree training, tuning, and interpretation
4. Rule-based classification
5. kNN experiments
6. Ensemble learning
7. Final comparison and recommendation

## Recommended Dataset Choice

Recommended default: **Titanic dataset**

Why this is the safest choice:

- Small and manageable for a graded assignment
- Clear binary classification target
- Includes missing values, categorical variables, and numeric features
- Easy to explain preprocessing decisions
- Works well for Logistic Regression, Decision Tree, kNN, Naive Bayes, and ensemble models
- Easier to interpret and present within a 6-page report

If we already have another dataset ready, we can adapt this plan, but Titanic is the fastest path to a strong submission.

## Deliverables We Should Create

- `assignment_notebook.ipynb`
- `report.pdf`
- `plan.md`
- Optional helper files:
  - `data/`
  - `figures/`
  - `outputs/`
  - `requirements.txt`
  - `README.md`

## Execution Plan

### Step 1: Set up the project structure

- Create folders for raw data, outputs, and figures
- Download and place the selected dataset into the repo
- Confirm target column and feature list
- Note dataset source for the report

### Step 2: Understand the dataset

- Describe the dataset shape, columns, target variable, and class distribution
- Identify attribute types:
  - Nominal
  - Ordinal
  - Numeric discrete
  - Numeric continuous
- Record obvious data quality issues:
  - Missing values
  - Duplicates
  - Outliers or noisy values

### Step 3: Preprocessing pipeline

- Remove or justify duplicates
- Handle missing values using clear reasoning
- Encode categorical variables
- Scale features where required
- Split into train and test sets

Important note:

- Scaling is essential for kNN
- Scaling may also help Logistic Regression
- Tree-based models do not usually require scaling

### Step 4: Train baseline models

Minimum baseline:

- Logistic Regression

Optional second baseline:

- Naive Bayes

For each baseline model, report:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC if applicable

### Step 5: Decision Tree task

- Train a Decision Tree classifier
- Tune:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- Extract and discuss feature importance
- Explain at least 2 decision paths if feasible

Recommended output:

- Tree plot or readable summary
- Short interpretation of why the tree makes certain splits

### Step 6: Rule-based classification

Preferred approach:

- Extract rules from the trained Decision Tree

Why:

- It satisfies the rule-based requirement without introducing risky external dependencies
- It supports interpretability discussion directly

Required output:

- Show at least 5 rules
- Explain interpretability vs performance tradeoff

### Step 7: kNN experiments

- Train kNN with `k = 1, 3, 5, 7, 9`
- Try multiple distance metrics if practical, such as:
  - Euclidean
  - Manhattan
- Compare results across k values

Required explanation:

- Why scaling is critical for kNN
- How k changes bias and variance

Recommended output:

- Small comparison table for k values
- One plot of performance vs k

### Step 8: Ensemble learning

Train at least 2 ensemble models:

- Random Forest
- Gradient Boosting

Optional:

- AdaBoost
- XGBoost if the environment supports it

Compare ensemble results against:

- Baseline model
- Decision Tree
- kNN

Recommended interpretation:

- Which ensemble performs best
- Whether the performance gain is worth the lower interpretability

### Step 9: Final model comparison

Create a final table with at least:

- Model name
- Accuracy
- F1-score
- Notes

Then answer clearly:

1. Which model performed best and why?
2. Which model is most interpretable and why?
3. What deployment risks would exist in real life?

## Recommended Notebook Structure

1. Title and objective
2. Dataset loading
3. Exploratory data analysis
4. Data quality assessment
5. Preprocessing
6. Baseline models
7. Decision Tree
8. Rule extraction
9. kNN experiments
10. Ensemble models
11. Final comparison
12. Conclusion

Notebook quality rules:

- Clean runnable cells
- Proper headings
- Outputs visible
- Graphs included
- Short markdown explanations before each major section

## Recommended Report Structure

Keep the report within 6 pages.

1. Introduction and dataset description
2. Data preprocessing decisions
3. Model setup and evaluation approach
4. Results and comparison tables
5. Key graphs
6. Final conclusion and recommendation

The report should focus on:

- Clear justification
- Compact tables
- Only the most useful visuals
- Strong final comparison

## Evaluation Strategy

To score well on the rubric, we should optimize for:

- Correct ML concepts from preprocessing through ensembles
- Strong reasoning for every preprocessing and model choice
- Clear organization in both notebook and report
- Original comparisons and thoughtful conclusions
- Clean, technically correct implementation
- Professional presentation

## Risks to Avoid

- Using too many models without clear interpretation
- Weak explanation of preprocessing choices
- Forgetting rule-based classification
- Forgetting the final comparison table
- Not showing at least 5 rules
- Running kNN without scaling
- Submitting a messy notebook with missing outputs
- Writing a report that exceeds 6 pages
- Copying content too closely from online notebooks

## Definition of Done

This assignment is ready when all items below are complete:

- One dataset chosen and documented
- Missing values, duplicates, and outliers analyzed
- Attribute types explained
- Preprocessing completed and justified
- Train/test split completed
- Logistic Regression trained and evaluated
- Decision Tree trained, tuned, and interpreted
- Rule-based classification shown with at least 5 rules
- kNN tested for 1, 3, 5, 7, and 9 neighbors
- At least 2 ensemble models trained and compared
- Final model comparison table created
- Best model, most interpretable model, and deployment risks discussed
- Notebook is clean and runnable
- Report is polished and within 6 pages
- Final zip file prepared for upload

## Suggested Next Move

Best next step: choose the dataset and scaffold the notebook immediately.

If we want the fastest path, we should proceed with Titanic and then build the notebook section by section from this plan.
