import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
INPUT_FILE = './Datasets/hyperlink_features_with_pct.csv'
MODEL_OUTPUT_FILE = './model/rf_hyperlink_model.pkl'
PREDICTIONS_OUTPUT_FILE = './model/rf_hyperlink_predictions.csv'

try:
    data = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found.")
    exit()

data.dropna(inplace=True)
feature_cols = [col for col in data.columns if col not in ['label', 'url']]
print(f"Found {len(feature_cols)} features.")
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)
X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']
print(f"Training set: {len(X_train)} samples")
print(f"Testing set:  {len(X_test)} samples")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model training complete!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
if cm.shape == (2, 2):
    print(f"False Negatives (Phishing marked as Safe): {cm[1][0]}")

print("\nClassification Report:")
report = classification_report(
    y_test, 
    y_pred, 
    target_names=['Safe (0)', 'Phishing (1)'],
    zero_division=0
)
print(report)

print(f"\nSaving trained model to '{MODEL_OUTPUT_FILE}'")
with open(MODEL_OUTPUT_FILE, 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully (using pickle).")

print(f"Saving predictions to '{PREDICTIONS_OUTPUT_FILE}'")
df_predictions = pd.DataFrame({
    'url': test_df['url'],
    'actual_label': test_df['label'],
    'predicted_label': y_pred
})

df_predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)
print("Predictions saved successfully.")

