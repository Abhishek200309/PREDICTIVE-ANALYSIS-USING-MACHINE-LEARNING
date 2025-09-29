# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: ABHISHEK VICTOR RAJ MANESH

*INTERN ID*: CT12DY2725

*DOMAIN*: DATA ANALYTICS

*DURATION*: 12 WEEKS

*MENTOR*: NEELA SANTOSH

# Description

This project applies machine learning techniques to the classic Titanic dataset, which records details about passengers on the Titanic and whether they survived. The main objective is to build a classification model that predicts survival based on passenger attributes such as age, sex, ticket class, and fare.

The project demonstrates the end-to-end ML workflow:

Data preprocessing and feature engineering

Model training with hyperparameter tuning

Model evaluation using accuracy, precision, recall, F1-score

Visualization of performance with confusion matrix and bar charts

Automated HTML reporting for easy interpretation

By following this pipeline, we gain practical experience in data cleaning, model building, evaluation, and reporting, which are essential skills in real-world data science projects.

# Tools & Libraries Used

Python 3.x – Core programming language

Pandas – For data handling and preprocessing

NumPy – For numerical computations

Scikit-learn – For ML models and evaluation metrics

Matplotlib & Seaborn – For visualizations

io & base64 – To embed images directly in HTML reports

Webbrowser – To open the generated report automatically

# Workflow Explanation
1. Data Loading

We begin by loading the Titanic dataset (CSV format) into a Pandas DataFrame. This dataset contains features like Pclass (ticket class), Sex, Age, Fare, and Embarked, along with the target variable Survived.

2. Data Preprocessing & Feature Engineering

Since machine learning algorithms work best with numeric, clean data, we perform several preprocessing steps:

Dropping rows with missing survival labels.

Handling missing values: replacing missing Age with the median and missing Embarked values with the mode.

Encoding categorical features (Sex and Embarked) into numeric values using LabelEncoder.

Scaling numerical features with StandardScaler to ensure all features have similar ranges.

3. Splitting Data

We split the dataset into training (80%) and testing (20%) sets. The training set is used to train the model, while the testing set evaluates its ability to generalize to unseen data.

4. Model Training (Logistic Regression with Grid Search)

We use Logistic Regression, a well-known model for binary classification. To improve performance, we apply GridSearchCV to find the best regularization parameter (C). Cross-validation ensures that the chosen parameter works well across different subsets of the training data.

5. Evaluation

The model’s predictions are compared to actual labels in the test set. We calculate:

Accuracy – Percentage of correct predictions.

Precision – Correctness of positive (Survived) predictions.

Recall – Ability to detect all actual survivors.

F1-Score – Balance between precision and recall.

We also generate a classification report and confusion matrix to provide a detailed breakdown of model performance.

6. Visualization

To make results easier to interpret, we include:

Confusion Matrix Heatmap – Shows correct vs incorrect predictions.

Class Distribution Plot – Displays balance of classes in the dataset.

Performance Metrics Chart – Bar chart comparing precision, recall, and F1-score.

7. Automated HTML Report

All results (accuracy, best parameters, metrics, and charts) are compiled into an HTML report. Plots are embedded directly into the file using base64 encoding, so the report is fully portable with no external dependencies.

# Results & Insights

Logistic Regression achieved solid accuracy on the Titanic dataset.

Sex proved to be a strong predictor of survival, with women having a much higher survival rate.

The confusion matrix revealed that while the model was good at identifying non-survivors, it sometimes misclassified survivors.

The generated HTML report provides a user-friendly summary that combines text-based metrics with visuals, making evaluation intuitive.

# Key Learnings

Data preprocessing is crucial – Handling missing values, scaling, and encoding categorical variables significantly impact model performance.

Evaluation beyond accuracy – Metrics like precision, recall, and F1-score help understand strengths and weaknesses in different classes.

Automation improves reproducibility – Automating reporting ensures results are shareable and easy to interpret.

Logistic Regression is interpretable – It provides a simple yet effective baseline for binary classification problems.

<img width="1357" height="834" alt="Image" src="https://github.com/user-attachments/assets/0f4d42db-5345-4d78-9d91-660418b55572" />
<img width="1335" height="578" alt="Image" src="https://github.com/user-attachments/assets/cabb485c-87d4-4adb-9d99-5f0e00cb9ba7" />
<img width="1261" height="516" alt="Image" src="https://github.com/user-attachments/assets/ecc8ac97-dfba-4b71-9357-7bf1096e7443" />
<img width="1241" height="475" alt="Image" src="https://github.com/user-attachments/assets/6e7ff618-f9ae-4efe-b008-29986bcc65c4" />

# Conclusion

This project showcases a complete machine learning pipeline applied to the Titanic dataset, from raw data to an automatically generated HTML report. By walking through preprocessing, training, evaluation, and visualization, it provides a strong foundation for beginners in data science and machine learning.

The approach is not only practical for the Titanic dataset but can also be applied to other binary classification problems in real-world scenarios.
