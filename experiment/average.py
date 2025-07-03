import os
import pandas as pd
import numpy as np
import re
import csv

def average():
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

def average_transfer(folder_path):
    total = 0.0
    count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if "Average Rank" in row:
                        val = row["Average Rank"]
                        try:
                            mean_str = val.split("±")[0].strip()
                            total += float(mean_str)
                            count += 1
                        except (ValueError, IndexError):
                            pass  # skip if format is wrong

    if count == 0:
        raise ValueError("No valid 'Average Rank' found in the CSV files.")

    average_rank = total / count

    # save to a new CSV file
    with open("average_transfer.csv", "w", newline='') as csvfile:
        fieldnames = ["base_model", "transfer_model", "Average Rank"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"base_model": "llama-3.1-8b",
                        "transfer_model": "gpt-4.1",
                        "Average Rank": f"{average_rank:.2f}"})

if __name__ == "__main__":
    average_transfer("transfer/ragroll/llama-3.1-8b")
