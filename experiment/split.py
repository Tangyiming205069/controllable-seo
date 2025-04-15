from datasets import load_dataset
import transformers
import json, os

def split_amazon_dataset(catalog, version, num_samples=10):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{catalog}", split="full", trust_remote_code=True)
    checker_tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                    trust_remote_code=True,
                                                    use_fast=False,
                                                    use_cache=True)
    def filter_fn(example):
        if example['description'] == []:
            return False
        description = example['description'][0]
        title = example['title']
        description_token = checker_tokenizer(description, return_tensors='pt', add_special_tokens=False)
        title_token = checker_tokenizer(title, return_tensors='pt', add_special_tokens=False)
        # return 30 <= description_token['input_ids'].shape[1] <= 50 and title_token['input_ids'].shape[1] <= 20
        return 40 <= description_token['input_ids'].shape[1] + title_token['input_ids'].shape[1] <= 60

    filtered_dataset = dataset.filter(filter_fn, num_proc=1)

    # random select 10 
    filtered_dataset = filtered_dataset.shuffle().select(range(num_samples))

    product_list = [
            {"Name": example["title"], "Natural": f"{example['title']}. Description: {example['description'][0]}"}
            for example in filtered_dataset
        ]
    os.makedirs(f"data2/amazon/{catalog}", exist_ok=True)
    with open(f"data2/amazon/{catalog}/{version}.jsonl", "w") as f:
        for item in product_list:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    for v in range(1, 101):
        split_amazon_dataset("All_Beauty", v, num_samples=10)