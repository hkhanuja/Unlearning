import ast
import os
import pandas as pd

from config import ORIGINAL_DATA_FILE, CLASSIFICATION_DATA_DIR, SAFE_DATA_FILE

original_df = pd.read_csv(ORIGINAL_DATA_FILE)
safe_df = pd.concat([original_df[original_df['is_response_0_safe']][['prompt', 'response_0']].rename(columns={'response_0': 'response'}), 
                     original_df[original_df['is_response_1_safe']][['prompt', 'response_1']].rename(columns={'response_1': 'response'})])
safe_df.to_csv(SAFE_DATA_FILE, index=False)

# Converting strings to dictionaries
for i in range(2):
    original_df[f'response_{i}_harm_category'] = original_df[f'response_{i}_harm_category'].apply(ast.literal_eval)

safety_categories = list(original_df['response_0_harm_category'].iloc[0].keys())
for cat in safety_categories:
    df = pd.DataFrame(columns=['prompt', 'response'])
    df.to_csv(os.path.join(CLASSIFICATION_DATA_DIR, f'{cat}.csv'), index=False)

for i in range(2):
    unsafe_df = original_df[~original_df[f'is_response_{i}_safe']]
    print(unsafe_df[['prompt', 'response_0', 'response_1']])
    unsafe_df = pd.concat([unsafe_df[['prompt', f'response_{i}']].rename(columns={f'response_{i}': 'response'}).reset_index(drop=True), 
                           pd.json_normalize(unsafe_df[f'response_{i}_harm_category']).reset_index(drop=True)
                           ], axis=1)
    
    for cat in safety_categories:
        unsafe_cat_df = unsafe_df[unsafe_df[cat].astype(bool)][['prompt', 'response']].reset_index(drop=True)
        unsafe_cat_df.to_csv(os.path.join(CLASSIFICATION_DATA_DIR, f'{cat}.csv'), mode='a', header=False, index=False)
