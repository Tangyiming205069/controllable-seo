import os
import pandas as pd
import numpy as np
import re

folder_path = "metric"
pattern = re.compile(r'([+-]?\d*\.?\d+)\s*±\s*([+-]?\d*\.?\d+)')

sum_values = None
sum_squared_errors = None
file_count = 0

for file in os.listdir(folder_path):
    if file.endswith('.csv') or file.endswith('.xlsx') or file.endswith('.xls'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_excel(file_path)
        df = df.set_index("Model") if "Model" in df.columns else df.set_index(df.columns[0])
        df_numeric = df.iloc[:, 1:]
        values = df_numeric.apply(lambda col: col.map(lambda x: float(pattern.match(str(x)).group(1)) if pattern.match(str(x)) else np.nan))
        errors = df_numeric.apply(lambda col: col.map(lambda x: float(pattern.match(str(x)).group(2)) if pattern.match(str(x)) else np.nan))
        if sum_values is None:
            sum_values = values
            sum_squared_errors = errors ** 2
        else:
            sum_values += values
            sum_squared_errors += errors ** 2
        file_count += 1

avg_values = sum_values / file_count
avg_errors = (sum_squared_errors / file_count) ** 0.5
result_df = avg_values.round(3).astype(str) + " ± " + avg_errors.round(3).astype(str)
print(result_df)
print('file counts:',file_count)
result_df.to_excel("metrics.xlsx")
