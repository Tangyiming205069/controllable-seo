import pandas as pd
import os, torch
from tabulate import tabulate
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiment.get import get_model


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


def calculate_perplexity(text, model, tokenizer, device):
    input_ids = tokenizer(text, padding=True, return_tensors='pt')['input_ids'].to(device)

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
        loss = output.loss

    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_avg_perplexity(result_dir, model, catalog, random_inference, ppl_model, tokenizer, device, indices=[1,2,3,4,5,6,7,8,9,10]):
    perplexities = []

    for idx in indices:
        file_path = f"{result_dir}/{model}/{catalog}/{idx}/random_inference={random_inference}.csv"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        min_rank = min(df['product_rank'].tolist())
        row = df[df['product_rank'] == min_rank]
        filtered_df = df[df['iter'] != 0] # not use the first iteration one

        current_perplexities = []
        for attack in filtered_df['attack_prompt']:
            attack_prompt = attack.lstrip('<span style="color:red;">').rstrip('</span>')
            current_perplexities.append(calculate_perplexity(attack_prompt, ppl_model, tokenizer, device))

        min_perplexity = min(current_perplexities)

        perplexities.append(min_perplexity)

    assert len(perplexities) > 0, f"No results found for {model}, {catalog}, random_inference={random_inference}"

    average_perplexity = sum(perplexities) / len(perplexities)
    std = statistics.stdev(perplexities)
    
    return average_perplexity, std


if __name__ == "__main__":
    results = []

    result_dir = "result/suffix/v2"  # Adjust this to your actual results directory

    models = ['vicuna-7b', 'llama-3.1-8b', 'mistral-7b', 'deepseek-7b']
    catalogs = ['books', 'coffee_machines', 'cameras']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer (GPT-2) for calculating perplexity
    perplexity_model, perplexity_tokenizer = get_model("lmsys/vicuna-7b-v1.5", 16, device)

    for model in models:
        for catalog in catalogs:
            for random_inference in [True, False]:
                avg_rank, std_rank = calculate_average_rank(result_dir, model, catalog, random_inference)
                avg_perplexity, std_perplexity = calculate_avg_perplexity(result_dir, model, catalog, random_inference, perplexity_model, perplexity_tokenizer, device)
                if avg_rank is not None:
                    results.append({
                        "Model": model,
                        "Catalog": catalog,
                        "Random Inference": random_inference,
                        "Average Rank": f'{round(avg_rank, 2)}±{round(std_rank, 2)}', 
                        "Average Perplexity": f'{round(avg_perplexity, 2)}±{round(std_perplexity, 2)}'
                    })
    # Convert results to DataFrame and print nicely formatted table
    df_results = pd.DataFrame(results)
    print(tabulate(df_results, headers='keys', tablefmt='grid'))

