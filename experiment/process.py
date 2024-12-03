import json, torch


def init_prompt(model, tokenizer, product_list, helper_tokens, target_product_idx, temperature, prompt_length, batch_size, device):
    target_product_str = json.dumps(product_list[target_product_idx])#[:-3]
    target_product_tokens = tokenizer(target_product_str, return_tensors="pt")["input_ids"]
    product_repeated = target_product_tokens.repeat(batch_size, 1).to(device)

    product_helper = torch.cat([product_repeated, helper_tokens], dim=1)

    output = model.generate(product_helper, max_length=prompt_length + product_helper.shape[-1], do_sample=True, top_k=10, 
                            attention_mask=torch.ones_like(product_helper).to(device),
                            pad_token_id=tokenizer.eos_token_id)
    
    logits = model(output).logits
    prompt_logits = logits[:, -(prompt_length+1):-1, :] #/ temperature

    return prompt_logits.to(torch.float32)


def process_headtail(tokenizer, system_prompt, product_list, user_msg, target_product, batch_size, device):
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
            head += json.dumps(product) 
            tail += "\n"
            # tail += head[-3:]
            # head = head[:-3]
        else:
            tail += json.dumps(product) + "\n"
    tail += "\n" + user_msg + " [/INST]"

    head_tokens = tokenizer(head, return_tensors="pt")["input_ids"].repeat(batch_size, 1).to(device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)

    return head_tokens, tail_tokens,


def process_text(tokenizer, target_str, batch_size, device):
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

def select_topk(logits, topk):
    topk_indices = torch.topk(logits, topk, dim=-1).indices  # Shape: (batch_size, length, topk)

    # Create a mask for the top-k indices
    topk_mask = torch.zeros_like(logits, dtype=torch.bool)  # Shape: (batch_size, length, vocab_size)
    topk_mask.scatter_(-1, topk_indices, True)  # Mark top-k indices as True   

    return topk_mask


def create_bad_words_mask(bad_words, logits):
    batch_size, length, vocab_size = logits.size()
    bad_word_mask = torch.zeros((vocab_size,), dtype=torch.bool, device=logits.device)
    bad_word_mask.scatter_(0, bad_words.flatten(), True)  # Mark bad word tokens as True

    # Expand bad_word_mask to match logits shape
    bad_word_mask = bad_word_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, vocab_size)
    bad_word_mask = bad_word_mask.expand(batch_size, length, vocab_size)  # Broadcast across batch and length

    return bad_word_mask


def mask_logits(logits, topk_mask, bad_word_mask):
    BIG_CONST = -1e5
    combined_mask = topk_mask | bad_word_mask  # Logical OR to keep top-k and bad word logits

    # Step 4: Apply the combined mask to logits
    # -65504 is the minimum value for half precision floating point
    masked_logits = logits + (~combined_mask * BIG_CONST)
    return masked_logits
