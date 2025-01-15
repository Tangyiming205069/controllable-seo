import transformers, torch, json

def get_user_query(user_msg_type, catalog):
    if catalog == "coffee_machines":
        catalog = "data/coffee_machines.jsonl"
        if user_msg_type == "default":
            user_msg = "I am looking for a coffee machine. Can I get some recommendations?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations?"
    elif catalog == "books":
        catalog = "data/books.jsonl"
        if user_msg_type == "default":
            user_msg = "I am looking for a book. Can I get some recommendations?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for a good adventure novel. Can I get some recommendations?"
    elif catalog == "cameras":
        catalog = "data/cameras.jsonl"
        if user_msg_type == "default":
            user_msg = "I am looking for a camera. Can I get some recommendations?"
        elif user_msg_type == "custom":
            user_msg = "I am looking for a high resolution camera. Can I get some recommendations?"
    elif catalog == "election_articles":
        catalog = "data/election_articles.jsonl"
        if user_msg_type == "default":
            user_msg = "I am looking for an article. Can I get some recommendations?"
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


def get_product_list(catalog, target_product_idx):
    product_list = []
    # TODO
    with open(f'data2/{catalog}.jsonl', "r") as file:
        for line in file:
            product_list.append(json.loads(line))

    target_product_idx = target_product_idx - 1
    
    #product_names = [product['Name'] for product in product_list]

    target_product = product_list[target_product_idx]['Name']

    target_str = "1. " + target_product
    return product_list, target_product, target_str
