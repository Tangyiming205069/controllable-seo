import transformers, torch, json

def get_user_query(catalog):
    # if catalog == "coffee_machines":
    #     if user_msg_type == "default":
    #         user_msg = "I am looking for a coffee machine. Can I get some recommendations from the following options?"
    #     elif user_msg_type == "custom":
    #         user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations from the following products?"
    # elif catalog == "books":
    #     if user_msg_type == "default":
    #         user_msg = "I am looking for a book. Can I get some recommendations from the following options?"
    #     elif user_msg_type == "custom":
    #         user_msg = "I am looking for a good adventure novel. Can I get some recommendations from the following books?"
    # elif catalog == "cameras":
    #     if user_msg_type == "default":
    #         user_msg = "I am looking for a camera. Can I get some recommendations from the following options?"
    #     elif user_msg_type == "custom":
    #         user_msg = "I am looking for a high resolution camera. Can I get some recommendations from the following cameras?"
    # elif catalog == "election_articles":
    #     if user_msg_type == "default":
    #         user_msg = "I am looking for an article. Can I get some recommendations from the following articles?"
    # elif catalog.split('/')[0] == 'All_Beauty':
    #     if user_msg_type == "default":
    #         user_msg = "I am looking for a beauty product. Can I get some recommendations from the following options?"
    # else:
    #     raise ValueError("Invalid catalog.")
    
    user_msg = f'I am looking for a {catalog}. Can I get some recommendations from the following products?'
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
    product_list = []
    with open(f'data2/{dataset}/{catalog}.jsonl', "r") as file:
        for line in file:
            product_list.append(json.loads(line))

    target_product_idx = target_product_idx - 1
    
    target_product = product_list[target_product_idx]['Name']

    target_product_natural = product_list[target_product_idx]['Natural']

    target_str = "1. " + target_product

    return product_list, target_product, target_product_natural, target_str
