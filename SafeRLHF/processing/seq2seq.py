import os
import pandas as pd

from SafeRLHF.config import ORIGINAL_DATA_FILE, SEQ2SEQ_DATA_DIR
from SafeRLHF.processing.classification import (
    convert_category_strings_to_dicts)


def main():
    original_df = pd.read_csv(ORIGINAL_DATA_FILE)
    original_df = original_df[original_df['is_response_0_safe']
                              != original_df['is_response_1_safe']]
    convert_category_strings_to_dicts(original_df)

    safety_categories = list(
        original_df['response_0_harm_category'].iloc[0].keys())

    for i in range(2):
        unsafe_i_df = original_df[~original_df[f'is_response_{i}_safe']]

        unsafe_i_df = pd.concat([(unsafe_i_df[['prompt', f'response_{i}', f'response_{1 - i}']]
                                .rename(columns={f'response_{i}': 'unsafe response',
                                                 f'response_{1 - i}': 'safe response'})
                                .reset_index(drop=True)),
                                (pd.json_normalize(unsafe_i_df[f'response_{i}_harm_category'])
                                .reset_index(drop=True))
                                 ], axis=1)

        for cat in safety_categories:
            unsafe_i_cat_df = (unsafe_i_df[unsafe_i_df[cat].astype(bool)]
                               [['prompt', 'unsafe response', 'safe response']]
                               .dropna().reset_index(drop=True))
            unsafe_i_cat_df.to_csv(os.path.join(SEQ2SEQ_DATA_DIR, f'{cat}.csv'),
                                   index=False)


if __name__ == '__main__':
    main()
