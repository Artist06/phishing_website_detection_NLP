import pandas as pd
import numpy as np
import pickle
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

INPUT_FILE = './input_data/input_data.pkl'
COMPARISON_CSV = './model/content_model_comparison.csv'
BEST_MODEL_FILE = './model/best_content_model.pkl'
PREDICTIONS_CSV = './model/content_model_predictions.csv'

print(f"Loading data from '{INPUT_FILE}'...")
try:
    with open(INPUT_FILE, 'rb') as input_file:
        input_data = pickle.load(input_file)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

print("Data loaded. Pre-processing...")

data = pd.DataFrame(input_data, columns=['index', 'url', 'page_title_vector', 'page_content_vector', 'label'])
data.drop(columns=['index'], inplace=True)

train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

try:
    title_train = np.stack(train_df['page_title_vector'].values)
    content_train = np.stack(train_df['page_content_vector'].values)
    X_train = np.concatenate([title_train, content_train], axis=1)
    y_train = train_df['label'].values
except Exception as e:
    print(f"Error stacking training vectors: {e}")
    exit()

try:
    title_test = np.stack(test_df['page_title_vector'].values)
    content_test = np.stack(test_df['page_content_vector'].values)
    X_test = np.concatenate([title_test, content_test], axis=1)
    y_test = test_df['label'].values
except Exception as e:
    print(f"Error stacking test vectors: {e}")
    exit()

print(f"Features created (X_train shape): {X_train.shape}")
print(f"Features created (X_test shape): {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train_scaled)} samples")
print(f"Testing set:  {len(X_test_scaled)} samples")

models_to_test = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Neural Net (MLPClassifier)": MLPClassifier(random_state=42, max_iter=500, early_stopping=True, hidden_layer_sizes=(50, 25))
}

results = {}
trained_models = {}

for name, model in models_to_test.items():
    print("\n" + "="*50)
    print(f"Testing Model: {name}")
    print("="*50)
    
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    trained_models[name] = models_to_test
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)'], output_dict=True, zero_division=0)

    print(f"Training & Prediction Time: {end_time - start_time:.2f} seconds")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    if cm.shape == (2,2):
        print(f"False Negatives (Phishing marked as Safe): {cm[1][0]}")

    phishing_report = report_dict['Phishing (1)']
    results[name] = {
        'Accuracy (%)': accuracy * 100,
        'Precision (%)': phishing_report['precision'] * 100,
        'Recall (TP Rate) (%)': phishing_report['recall'] * 100,
        'F1-Measure (%)': phishing_report['f1-score'] * 100,
        'False Negatives': cm[1][0] if cm.shape == (2,2) else 'N/A',
        'Time (sec)': end_time - start_time
    }

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

df_results = pd.DataFrame(results).T
df_results = df_results.sort_values(by='F1-Measure (%)', ascending=False)
df_results.index.name = "Model"

print(df_results)
df_results.to_csv(COMPARISON_CSV)
print(f"\nComparison report saved to '{COMPARISON_CSV}'")

best_model_name = df_results.index[0]
best_model = trained_models[best_model_name]
print(f"\nIdentified '{best_model_name}' as the best model.")
with open(BEST_MODEL_FILE, 'wb') as f:
    pickle.dump(best_model, f)
print("Best model saved successfully.")
y_pred_best = best_model.predict(X_test_scaled)
df_predictions = pd.DataFrame({
    'url': test_df['url'],
    'actual_label': test_df['label'],
    'predicted_label': y_pred_best
})
df_predictions.to_csv(PREDICTIONS_CSV, index=False)
