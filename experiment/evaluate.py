import pandas as pd
import os
from tabulate import tabulate
import statistics


def calculate_average_rank(result_dir, model, catalog, random_inference, indices=[1,2,3,4,5,6,7,8,9,10]):
    ranks = []

    for idx in indices:
        file_path = f"{result_dir}/{model}/{catalog}/{idx}/random_inference={random_inference}.csv"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        # take the last 5 rows
        df = df.tail(5) # last batch
        ranks.append(min(df['product_rank'].tolist()))

    assert len(ranks) > 0, f"No results found for {model}, {catalog}, random_inference={random_inference}"

    average_rank = sum(ranks) / len(ranks)
    std = statistics.stdev(ranks)

    return average_rank, std



if __name__ == "__main__":
    results = []

    result_dir = "result/suffix/v1"  # Adjust this to your actual results directory

    models = ['vicuna-7b', 'llama-3.1-8b', 'mistral-7b', 'deepseek-7b']
    catalogs = ['books', 'coffee_machines', 'cameras']

    for model in models:
        for catalog in catalogs:
            for random_inference in [True, False]:
                avg_rank, std = calculate_average_rank(result_dir, model, catalog, random_inference)
                if avg_rank is not None:
                    results.append({
                        "Model": model,
                        "Catalog": catalog,
                        "Random Inference": random_inference,
                        "Average Rank": round(avg_rank, 2), 
                        "Std": round(std, 2)
                    })
    # Convert results to DataFrame and print nicely formatted table
    df_results = pd.DataFrame(results)
    print(tabulate(df_results, headers='keys', tablefmt='grid'))

