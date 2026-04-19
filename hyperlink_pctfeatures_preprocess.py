import pandas as pd
import numpy as np

input_file = './Datasets/hyperlink_data.csv'
output_file = './Datasets/hyperlink_features_with_pct.csv'

try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded '{input_file}' with {len(df)} rows.")
    feature_pairs = {
        'pct_internal_links': ('internal_links', 'external_links'),
        'pct_internal_css': ('internal_css', 'external_css'),
        'pct_internal_favicon': ('internal_favicon', 'external_favicon')
    }
    for new_col_name, (internal_col, external_col) in feature_pairs.items():
        
        if internal_col in df.columns and external_col in df.columns:
            total_links = df[internal_col] + df[external_col]
            percentage = df[internal_col] / total_links.replace(0, np.nan)
            df[new_col_name] = percentage.fillna(0)
            
            print(f"  - Created column: '{new_col_name}'")
            
        else:
            print(f"  - Warning: Columns '{internal_col}' or '{external_col}' not found. Skipping '{new_col_name}'.")
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully saved new file to '{output_file}'")
    print("Here is a preview of your new data:")
    preview_cols = [
        'url', 'label', 
        'internal_links', 'external_links', 'pct_internal_links',
        'internal_css', 'external_css', 'pct_internal_css'
    ]
    print(df[preview_cols].head())

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure it's in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
