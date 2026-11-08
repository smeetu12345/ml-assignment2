# ShopperSense â€“ Predicting Online Purchase Intent (ML Assignment 2)

**Live app:** https://shoppersense-smeet.streamlit.app/  
**GitHub repo:** https://github.com/smeetu12345/ml-assignment2

## a. Problem statement

An e-commerce site wants to know, while a visitor is still browsing, whether that session will end in a purchase. This is set up as a **binary classification** problem: from the session's page-visit counts, time spent, bounce and exit rates, page value and visitor or calendar information, predict whether `Revenue` is True (the visitor bought something) or False.

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. They are then served through an interactive Streamlit app.

## b. Dataset description

| Item | Details |
|---|---|
| Source | UCI Machine Learning Repository â€“ [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (Sakar et al., 2018) |
| Instances | 12,330 sessions (12,205 after removing 125 duplicate rows) |
| Features | 17 input features plus 1 target |
| Target | `Revenue` (1 = purchase, 0 = no purchase) |
| Class balance | 84.4 % no purchase / 15.6 % purchase (imbalanced) |

**Features**
- *Numerical (10):* `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` (pages visited of each type and the time spent on them), `BounceRates`, `ExitRates`, `PageValues` (Google Analytics metrics), and `SpecialDay` (how close the visit is to a holiday).
- *Categorical (6):* `Month`, `VisitorType` (New / Returning / Other), `OperatingSystems`, `Browser`, `Region`, `TrafficType`.
- *Boolean (1):* `Weekend`.

**Preprocessing:** duplicate rows were dropped, numerical features were standardised, categorical features were one-hot encoded and `Weekend` was cast to 0/1. All of this is done inside a scikit-learn `Pipeline`, so each saved model takes raw rows as input. The data was split 80/20 with stratification on the target (`random_state=26`). The 20 % hold-out set (2,441 rows) is saved as `test_data.csv`. Because purchases are only about 15 % of sessions, `class_weight="balanced"` is used for LR, DT and SVM.

## c. GitHub repository

https://github.com/smeetu12345/ml-assignment2

```
â”œâ”€â”€ app.py                      # Streamlit app
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ test_data.csv               # hold-out test set (upload this in the app)
â”œâ”€â”€ data/online_shoppers_intention.csv   # raw UCI data
â”œâ”€â”€ ML_Assignment2_BITS_Lab.ipynb        # notebook run on BITS Virtual Lab
â””â”€â”€ model/
    â”œâ”€â”€ train_models.py         # training + evaluation script
    â”œâ”€â”€ metrics_summary.csv
    â”œâ”€â”€ logistic_regression.joblib
    â”œâ”€â”€ decision_tree.joblib
    â”œâ”€â”€ knn.joblib
    â”œâ”€â”€ naive_bayes.joblib
    â””â”€â”€ support_vector_machine.joblib
```

**Run locally**
```bash
pip install -r requirements.txt
python model/train_models.py     # retrains models and regenerates test_data.csv
streamlit run app.py
```

## d. Models used and comparison

All metrics are computed on the 20 % hold-out test set. Precision, Recall and F1 are for the positive class (purchase).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8566 | 0.9165 | 0.5278 | 0.7958 | 0.6347 | 0.5679 |
| Decision Tree | 0.8664 | **0.9378** | 0.5458 | **0.8743** | **0.6720** | **0.6197** |
| KNN | **0.8746** | 0.8378 | **0.6939** | 0.3560 | 0.4706 | 0.4370 |
| Naive Bayes | 0.8116 | 0.8280 | 0.4316 | 0.6440 | 0.5168 | 0.4179 |
| Support Vector Machine | 0.8714 | 0.9169 | 0.5630 | 0.7958 | 0.6594 | 0.5963 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline: AUC 0.92 and recall 0.80. With balanced class weights it finds most buyers, but precision is only 0.53, so it raises many false alarms. Its ranking ability is good, but a linear boundary cannot capture interactions such as PageValues Ã— ExitRates. |
| Decision Tree | Best model overall: highest AUC (0.938), recall (0.874), F1 (0.672) and MCC (0.620). The tree splits mainly on `PageValues` first, a threshold-style rule that suits this data. Limiting depth to 7 and requiring at least 20 samples per leaf stops it from overfitting. |
| KNN | Has the highest accuracy (0.875) and precision (0.694), but recall is only 0.356, so it misses about two-thirds of buyers. KNN cannot use class weights, so the majority "no purchase" class dominates the neighbourhoods. Its accuracy looks high only because of the class imbalance, and its MCC (0.437) is among the lowest. |
| Naive Bayes | Weakest model (MCC 0.418, accuracy 0.81). Its independence assumption is broken here because page counts and durations are strongly correlated, and the one-hot features are not Gaussian. Without extra variance smoothing its accuracy fell to about 0.30. |
| Support Vector Machine | Second best (MCC 0.596, F1 0.659). The RBF kernel captures non-linear patterns and gives better precision than LR at the same recall (0.80). It is the slowest model to train and needs feature scaling. |
| **Overall winner** | **Decision Tree.** It has the best MCC, F1, AUC and recall, which are the metrics that matter for an imbalanced business problem where finding buyers is the goal. SVM is a close second. Accuracy on its own is misleading here, as KNN shows. |

## Streamlit app features
- CSV upload of test data (sample `test_data.csv` downloadable from the sidebar)
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix, ROC curve and classification report
- Side-by-side comparison of all 5 models on the uploaded data

The models are only loaded and evaluated in the app. **No retraining** happens on uploaded data.
