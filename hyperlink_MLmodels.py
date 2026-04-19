import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

input_file = './Datasets/hyperlink_data.csv'
output_file = './model/hyperlink_model_comparison_withoutpct.csv'

try:
    data = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    exit()

data.dropna(inplace=True)
y = data['label']
X = data.drop(['label', 'url'], axis=1)
print(f"Using {X.shape[1]} features for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Testing set:  {len(X_test)} samples")

models_to_test = {
    "SMO (SVC-linear)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='linear', random_state=42))
    ]),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    
    "SVM (SVC-rbf)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', random_state=42))
    ]),
    
    "Adaboost": AdaBoostClassifier(random_state=42),
    
    "Neural Network (MLP)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(random_state=42, max_iter=500, early_stopping=True, hidden_layer_sizes=(50, 25)))
    ]),
    
    "C4.5 (Decision Tree)": DecisionTreeClassifier(criterion='entropy', random_state=42),
    
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
    ])
}
results_list = []

for name, model in models_to_test.items():
    print(f"Testing Model: {name}")

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Training & Prediction Time: {end_time - start_time:.2f} seconds")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"[tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}]")

    results_list.append({
        "Model": name,
        "True Positive (%)": tpr * 100,
        "True Negative(%)": tnr * 100,
        "Precision (%)": precision * 100,
        "f1 Measure (%)": f1 * 100,
        "Accuracy (%)": accuracy * 100
    })


print("\nFINAL MODEL COMPARISON")
df_results = pd.DataFrame(results_list)
df_results = df_results.sort_values(by='f1 Measure (%)', ascending=False)
df_results = df_results.set_index("Model")

print(df_results.to_string(float_format="%.2f"))
df_results.to_csv(output_file)
print(f"\nSuccessfully saved final model report to '{output_file}'")
best_model_name = df_results.index[0]
best_f1_score = df_results.iloc[0]['f1 Measure (%)']
print(f"\nBest Model (based on F1-Score): {best_model_name} (F1 = {best_f1_score:.2f}%)")
