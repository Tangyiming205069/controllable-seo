import transformers, torch, json
from datasets import load_dataset

def get_user_query(user_msg_type, catalog):
    if catalog == "coffee_machines":
        if user_msg_type == "default":
            user_msg = "I am looking for a coffee machine. Can I get some recommendations from the following options?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations from the following products?"
    elif catalog == "books":
        if user_msg_type == "default":
            user_msg = "I am looking for a book. Can I get some recommendations from the following options?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for a good adventure novel. Can I get some recommendations from the following books?"
    elif catalog == "cameras":
        if user_msg_type == "default":
            user_msg = "I am looking for a camera. Can I get some recommendations from the following options?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for a high resolution camera. Can I get some recommendations from the following cameras?"
    elif catalog == "election_articles":
        if user_msg_type == "default":
            user_msg = "I am looking for an article. Can I get some recommendations from the following articles?"
    elif catalog == 'All_Beauty':
        if user_msg_type == "default":
            user_msg = "I am looking for a beauty product. Can I get some recommendations from the following options?"
    else:
        raise ValueError("Invalid catalog.")
    return user_msg


def get_model(model_path, precision, device):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=eval(f'torch.float{precision}'),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False
    ).to(device).eval()

    model.generation_config.do_sample = True

    for param in model.parameters():
        param.requires_grad = False
 
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                           trust_remote_code=True,
                                                           use_fast=False,
                                                           use_cache=True)

    if 'llama' in model_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'

    return model, tokenizer


def get_product_list(catalog, target_product_idx, dataset):

    
    if dataset == 'json':
        product_list = []
        with open(f'data2/{catalog}.jsonl', "r") as file:
            for line in file:
                product_list.append(json.loads(line))

        target_product_idx = target_product_idx - 1

        #product_names = [product['Name'] for product in product_list]


    elif dataset == 'amazon':
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
        filtered_dataset = filtered_dataset.shuffle().select(range(10))

        product_list = [
                {"Name": example["title"], "Natural": f"{example['title']}. Description: {example['description'][0]}"}
                for example in filtered_dataset
            ]

    else:
        raise ValueError("Invalid dataset.")
    
    target_product = product_list[target_product_idx]['Name']

    target_product_natural = product_list[target_product_idx]['Natural']

    target_str = "1. " + target_product

    return product_list, target_product, target_product_natural, target_str
