# task2_ml_predictive_analysis_html.py
import time
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])
le_emb = LabelEncoder()
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 3) Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# 4) Model Training
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
accuracy = accuracy_score(y_test, y_pred)

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df = class_report_df.rename(index={
    '0': 'Died',
    '1': 'Survived',
    'accuracy': 'Overall Accuracy',
    'macro avg': 'Macro Average',
    'weighted avg': 'Weighted Average'
}).round(2)

# Confusion matrix
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
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died","Survived"], yticklabels=["Died","Survived"], ax=ax)
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

# Performance metrics bar chart
metrics = class_report_df.loc[['Died', 'Survived'], ['precision','recall','f1-score']]
fig_metrics, ax = plt.subplots(figsize=(6,4))
metrics.plot(kind='bar', ax=ax)
ax.set_ylim(0,1)
ax.set_title("Performance Metrics per Class")
ax.set_ylabel("Score")
ax.set_xticklabels(metrics.index, rotation=0)
ax.legend(loc="lower right")
metrics_b64 = plot_to_base64(fig_metrics)

# -------------------------
# 7) Collect results
# -------------------------
results = {
    "accuracy": accuracy,
    "best_params": grid.best_params_,
    "classification_report": class_report_df,
    "conf_matrix_img": cm_b64,
    "class_dist_img": dist_b64,
    "metrics_img": metrics_b64,
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
        .img-container {{ margin-bottom: 20px; text-align: center; }}
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

<h2>📈 Performance Metrics per Class</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['metrics_img']}" />
</div>

<h2>🌀 Confusion Matrix</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['conf_matrix_img']}" />
</div>

<h2>📊 Test Set Class Distribution</h2>
<div class="img-container">
<img src="data:image/png;base64,{results['class_dist_img']}" />
</div>

</body>
</html>
"""

report_path = "D:/197001__/titanic_ml_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

webbrowser.open(f"file://{report_path}")
print(f"✅ HTML report saved at {report_path}")
