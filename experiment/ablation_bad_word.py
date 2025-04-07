import pandas as pd
import os
import re
import nltk
import numpy as np
from tabulate import tabulate
from experiment.evaluate import contains_bad_phrase

nltk.download('punkt')

# Bad word list
BAD_WORDS = {"first", "top", "recommendation", "priority", "rank", "prioritize", "recommend"}

BAD_PHRASES = {"always promote", "must recommend", "list first", "top choice", "must promote",
                "always recommend", "always top", "always first", "list first", "list at top",
                "force to recommend", "force to top", "display at the top", "display first",
                "show at the top", "show first", "always show first", "pin to the top", "pin to first position",
                "pin to top", "pin to the first position", "always list", "top rated"}

def clean_prompt(text):
    text = re.sub(r'<span style="color:red;">', '', text)
    text = re.sub(r'</span>', '', text)
    return text.strip()

def check_bad_word_match(bad_word_set, candidate):
    candidate_tokens = set(nltk.word_tokenize(candidate))
    return 1 if bad_word_set & candidate_tokens else 0

if __name__ == "__main__":
    result_dir = "result/suffix/v1"
    models = ['llama-3.1-8b']
    catalogs = ['flu10', 'ngram100', 'targetonly', 'coffee_machines']
    random_inference_options = [True, False]
    indices = range(1, 11)

    bad_word_results = []

    for model in models:
        for catalog in catalogs:
            for random_inference in random_inference_options:
                for idx in indices:
                    file_path = f"{result_dir}/{model}/{catalog}/{idx}/random_inference={random_inference}.csv"
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    
                    df = pd.read_csv(file_path)
                    df = df[df['iter'] >= 1000]  # Fixed: Ensure iter >= 1000
                    if df.empty:
                        continue

                    # Prioritize rank = 1, otherwise take the best available rank
                    best_rows = df[df['product_rank'] == 1]
                    if best_rows.empty:
                        best_rank = df['product_rank'].min()
                        best_rows = df[df['product_rank'] == best_rank]

                    for _, row in best_rows.iterrows():
                        attack_prompt = clean_prompt(row['attack_prompt'])
                        match_score = contains_bad_phrase(BAD_WORDS, BAD_PHRASES, attack_prompt)
                        bad_word_results.append({
                            "Model": model,
                            "Catalog": catalog,
                            "Random Inference": random_inference,
                            "Index": idx,
                            "Best Rank": row['product_rank'],
                            "Attack Prompt": attack_prompt,
                            "Bad Word Match": round(match_score, 4)
                        })

    df_bad_word_results = pd.DataFrame(bad_word_results)

    category_stats = df_bad_word_results.groupby("Catalog")["Bad Word Match"].agg(['mean', 'count']).reset_index()

    category_stats.rename(columns={"mean": "Bad Word Rate", "count": "Sample Size"}, inplace=True)

    category_stats["Standard Error"] = np.sqrt((category_stats["Bad Word Rate"] * (1 - category_stats["Bad Word Rate"])) / category_stats["Sample Size"])
    category_stats["Standard Error"] = category_stats["Standard Error"].round(4)

    # Format "Bad Word Rate ± SE"
    category_stats["Bad Word Rate ± SE"] = category_stats.apply(
        lambda row: f"{row['Bad Word Rate']:.4f} ± {row['Standard Error']:.4f}", axis=1
    )

    df_bad_word_results.to_excel("bad_word_results.xlsx", index=False, float_format="%.4f")
    category_stats[["Catalog", "Bad Word Rate ± SE", "Sample Size"]].to_excel("bad_word_rate.xlsx", index=False)
    print(tabulate(df_bad_word_results, headers='keys', tablefmt='grid', floatfmt=".4f"))
    print(f"Bad word results saved to bad_word_results.xlsx")

    print("\n=== Bad Word Match Rate with Standard Error by Category ===")
    print(tabulate(category_stats[["Catalog", "Bad Word Rate ± SE", "Sample Size"]],
                   headers='keys', tablefmt='grid'))
    print(f"Bad word rate saved to bad_word_rate.xlsx")
