import os
import pandas as pd

from config import FINAL_DATA_DIR

methods = [d for d in os.listdir(FINAL_DATA_DIR)
           if os.path.isdir(os.path.join(FINAL_DATA_DIR, d))]

for method in methods:
    method_data_dir = os.path.join(FINAL_DATA_DIR, method)
    method_data_file_names = [file for file in os.listdir(method_data_dir)
                              if file.endswith('.csv')]

    for method_data_file_name in method_data_file_names:
        method_data_file = os.path.join(method_data_dir,
                                        method_data_file_name)
        data_df = pd.read_csv(method_data_file)
        data_df = data_df[['score', 'cohesion score',
                           'safer score', 'safer cohesion score',
                           'safe cohesion score']]
        for col in ['score', 'safer score']:
            data_df[col] = (data_df[col] > 0.5).astype(int)
        stats = data_df.mean()
        stats.to_csv(os.path.join(method_data_dir,
                     f'{method_data_file_name}_stats.csv'))
