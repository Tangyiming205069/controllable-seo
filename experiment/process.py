import json, torch


def init_prompt(model, tokenizer, product_list, target_product_idx, temperature, prompt_length, batch_size, device):
    target_product_str = json.dumps(product_list[target_product_idx])[:-3]
    target_product_tokens = tokenizer(target_product_str, return_tensors="pt")["input_ids"]
    product_repeated = target_product_tokens.repeat(batch_size, 1).to(device)
    output = model.generate(product_repeated, max_length=prompt_length + product_repeated.shape[-1], do_sample=True, top_k=10, 
                            attention_mask=torch.ones_like(product_repeated).to(device),
                            pad_token_id=tokenizer.eos_token_id)
    logits = model(output).logits
    prompt_logits = logits[:, -(prompt_length+1):-1, :] / temperature

    return prompt_logits


def proces_headtail(tokenizer, system_prompt, product_list, user_msg, target_product, batch_size, device):
    # since it might shuffle the product list, we need to find the index of the target product

    product_names = [product['Name'] for product in product_list]
    target_product_idx = product_names.index(target_product)

    head = system_prompt
    tail = ''

    # Generate the adversarial prompt
    for i, product in enumerate(product_list):
        if i < target_product_idx:
            head += json.dumps(product) + "\n"
        elif i == target_product_idx:
            head += json.dumps(product) + "\n"
            tail += head[-3:]
            head = head[:-3]
        else:
            tail += json.dumps(product) + "\n"
    tail += "\n" + user_msg + " [/INST]"

    head_tokens = tokenizer(head, return_tensors="pt")["input_ids"].repeat(batch_size, 1).to(device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)

    return head_tokens, tail_tokens,


def process_target(tokenizer, target_str, batch_size, device):
    target_tokens = tokenizer(target_str, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)
    return target_tokens


def process_bad_words(bad_words, tokenizer, device):
    upper_words = [word.upper() for word in bad_words]

    all_bad_words = bad_words + upper_words
    
    bad_words_str = ' '.join(all_bad_words)

    bad_words_ids = tokenizer(bad_words_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    return bad_words_ids

def greedy_decode(logits, tokenizer):
    token_ids = torch.argmax(logits, dim=-1) 

    decoded_sentences = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in token_ids]

    return token_ids, decoded_sentences
