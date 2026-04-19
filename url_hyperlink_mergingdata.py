import pandas as pd

FILE_A = './Datasets/url_features_extracted.csv'
FILE_B = './Datasets/hyperlink_features_with_pct.csv'
OUTPUT_FILE = './Datasets/final_master_features.csv'

try:
    df_url_features = pd.read_csv(FILE_A)
except FileNotFoundError:
    print(f"Error: The file '{FILE_A}' was not found.")
    exit()

try:
    df_hyperlink_features = pd.read_csv(FILE_B)
except FileNotFoundError:
    print(f"Error: The file '{FILE_B}' was not found.")
    exit()

print(f"\nLoaded {len(df_url_features)} rows from {FILE_A}")
print(f"Loaded {len(df_hyperlink_features)} rows from {FILE_B}")

print(f"\nMerging datasets on the 'url' column...")

df_hyperlink_features = df_hyperlink_features.drop('label', axis=1, errors='ignore')
merged_df = pd.merge(
    df_url_features, 
    df_hyperlink_features, 
    on='url', 
    how='inner'
)
print(f"Merge complete. The new master dataset has {len(merged_df)} rows.")

print(f"Saving final master feature set to '{OUTPUT_FILE}'")
merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"Final master dataset saved to '{OUTPUT_FILE}'.")
print(f"It contains {len(merged_df.columns)} total columns (features + url/label).")
