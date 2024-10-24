import ast
import os
import pandas as pd

from SafeRLHF.config import ORIGINAL_DATA_FILE, CLS_DATA_DIR, CLS_SAFE_DATA_FILE


def save_safe_data(df: pd.DataFrame, file_path: str) -> None:
    safe_df = pd.concat([(df[df['is_response_0_safe']][['prompt', 'response_0']]
                        .rename(columns={'response_0': 'response'})),
                        (df[df['is_response_1_safe']][['prompt', 'response_1']]
                        .rename(columns={'response_1': 'response'}))
                         ])
    safe_df.dropna(inplace=True)
    safe_df.to_csv(file_path, index=False)


def convert_category_strings_to_dicts(df: pd.DataFrame) -> None:
    for i in range(2):
        df[f'response_{i}_harm_category'] = df[f'response_{i}_harm_category'].apply(
            ast.literal_eval)


def create_safety_categories_files(categories: list, output_dir: str) -> None:
    '''
    Creates empty files for each category to fill later.
    '''
    for cat in categories:
        df = pd.DataFrame(columns=['prompt', 'response'])
        df.to_csv(os.path.join(output_dir, f'{cat}.csv'), index=False)


def save_unsafe_data_by_category(df: pd.DataFrame,
                                 i: int,
                                 categories: list,
                                 output_dir: str) -> None:
    unsafe_df = df[~df[f'is_response_{i}_safe']]
    # Making column of boolean values for each category
    # and concatenating them with prompt-response pairs
    unsafe_df = pd.concat([(unsafe_df[['prompt', f'response_{i}']]
                            .rename(columns={f'response_{i}': 'response'})
                            .reset_index(drop=True)),
                           (pd.json_normalize(unsafe_df[f'response_{i}_harm_category'])
                            .reset_index(drop=True))
                           ], axis=1)

    for cat in categories:
        unsafe_cat_df = (unsafe_df[unsafe_df[cat].astype(bool)]
                         [['prompt', 'response']].dropna()
                         .reset_index(drop=True))
        unsafe_cat_df.to_csv(os.path.join(output_dir, f'{cat}.csv'),
                             mode='a', header=False, index=False)


def main():
    original_df = pd.read_csv(ORIGINAL_DATA_FILE)
    save_safe_data(original_df, CLS_SAFE_DATA_FILE)
    convert_category_strings_to_dicts(original_df)

    safety_categories = list(
        original_df['response_0_harm_category'].iloc[0].keys())
    create_safety_categories_files(safety_categories, CLS_DATA_DIR)

    for i in range(2):
        save_unsafe_data_by_category(
            original_df, i, safety_categories, CLS_DATA_DIR)


if __name__ == '__main__':
    main()
