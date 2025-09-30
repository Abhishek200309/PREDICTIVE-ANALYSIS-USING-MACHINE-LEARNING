# task2_ml_predictive_analysis_html.py (Extended with More Analysis)
import time
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# -------------------------
# 1) Load Dataset
# -------------------------
file_path = "D:/197001__/titanic.csv"   # Update path if needed
df = pd.read_csv(file_path)

# -------------------------
# 2) Preprocessing & Feature Engineering
# -------------------------
df = df.dropna(subset=["Survived"])
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
df = df[features + ["Survived"]]
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical features
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])
le_emb = LabelEncoder()
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 3) Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# 4) Model Training with GridSearchCV
# -------------------------
start = time.time()
param_grid = {"C": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_jobs=1)
grid.fit(X_train, y_train)
duration = time.time() - start

best_model = grid.best_estimator_

# -------------------------
# 5) Evaluation
# -------------------------
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:,1]
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# -------------------------
# 6) Helper function: convert plot to base64
# -------------------------
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Confusion matrix plot
fig_cm, ax = plt.subplots(figsize=(4,3))
cmap = plt.cm.Blues
ax.matshow(conf_matrix, cmap=cmap, alpha=0.5)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i,j], va='center', ha='center')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
cm_b64 = plot_to_base64(fig_cm)

# Class distribution plot
fig_dist, ax = plt.subplots(figsize=(4,3))
y_test.value_counts().sort_index().plot(kind='bar', ax=ax, color=['#3498db','#e74c3c'])
ax.set_xticklabels(['Died','Survived'], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Test Set Class Distribution")
dist_b64 = plot_to_base64(fig_dist)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
fig_roc, ax = plt.subplots(figsize=(5,4))
ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax.plot([0,1],[0,1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
roc_b64 = plot_to_base64(fig_roc)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_prec = average_precision_score(y_test, y_pred_proba)
fig_pr, ax = plt.subplots(figsize=(5,4))
ax.plot(recall, precision, label=f"AP = {avg_prec:.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
pr_b64 = plot_to_base64(fig_pr)

# Feature Importance
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": best_model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

fig_feat, ax = plt.subplots(figsize=(6,4))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm", ax=ax)
ax.set_title("Feature Importance (Logistic Regression Coefficients)")
feat_b64 = plot_to_base64(fig_feat)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

fig_lc, ax = plt.subplots(figsize=(6,4))
ax.plot(train_sizes, train_mean, 'o-', label="Training score")
ax.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend()
lc_b64 = plot_to_base64(fig_lc)

# -------------------------
# 7) Collect results
# -------------------------
results = {
    "accuracy": accuracy,
    "best_params": grid.best_params_,
    "classification_report": pd.DataFrame(class_report).transpose(),
    "conf_matrix_img": cm_b64,
    "class_dist_img": dist_b64,
    "roc_img": roc_b64,
    "pr_img": pr_b64,
    "feat_img": feat_b64,
    "lc_img": lc_b64,
    "train_time": duration
}

# -------------------------
# 8) Generate HTML Report
# -------------------------
html_content = f"""
<html>
<head>
    <title>🚀 Titanic ML Predictive Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 70%; margin-bottom: 20px; }}
        table, th, td {{ border: 1px solid #ccc; padding: 8px; }}
        th {{ background: #2c3e50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .img-container {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
<h1>🚀 Titanic ML Predictive Analysis Report</h1>

<h2>⏱ Training Time</h2>
<p>{results['train_time']:.2f} seconds</p>

<h2>🎯 Accuracy</h2>
<p>{results['accuracy']:.2%}</p>

<h2>🔧 Best Parameters</h2>
<p>{results['best_params']}</p>

<h2>📊 Classification Report</h2>
{results['classification_report'].to_html()}

<h2>🌀 Confusion Matrix</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['conf_matrix_img']}" />
</div>

<h2>📊 Test Set Class Distribution</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['class_dist_img']}" />
</div>

<h2>🚦 ROC Curve</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['roc_img']}" />
</div>

<h2>📉 Precision-Recall Curve</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['pr_img']}" />
</div>

<h2>📌 Feature Importance</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['feat_img']}" />
</div>

<h2>📚 Learning Curve</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['lc_img']}" />
</div>

</body>
</html>
"""

report_path = "D:/197001__/titanic_ml_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

webbrowser.open(f"file://{report_path}")
print(f"✅ HTML report saved at {report_path}")
