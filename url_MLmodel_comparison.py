import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

csv_file = './Datasets/url_features_extracted.csv'

try:
    data = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit()

data.dropna(inplace=True)
y = data['label']
X = data.drop(['label', 'url'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,stratify=y)
print(f"Training set: {len(X_train)} samples")
print(f"Testing set:  {len(X_test)} samples")

models_to_test = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100,n_jobs=-1),

    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    
    "Support Vector Machine (SVC)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf'))
    ]),
    
    "Neural Net (MLP)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(max_iter=500, early_stopping=True, hidden_layer_sizes=(50, 25)))
    ])
}

results = {}
for name, model in models_to_test.items():
    print(f"Testing Model: {name}")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)'])
    print(f"Training & Prediction Time: {end_time - start_time:.2f} seconds")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"False Negatives (Phishing marked as Safe): {cm[1][0]}")
    print("\nClassification Report:")
    print(report)

    results[name] = {
        'accuracy': accuracy,
        'f1_phishing': float(report.split()[-7]),
        'false_negatives': cm[1][0],
        'time': end_time - start_time
    }

print("FINAL MODEL COMPARISON")

df_results = pd.DataFrame(results).T.sort_values(by='f1_phishing', ascending=False)
df_results.index.name = "Model"
print(df_results[['accuracy', 'f1_phishing', 'false_negatives', 'time']])
print("\n'f1_phishing' is the F1-score for the 'Phishing (1)' class.")
df_results.to_csv('model_comparison_report.csv')