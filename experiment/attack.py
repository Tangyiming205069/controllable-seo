import torch, random, wandb, unicodedata
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from experiment.process import greedy_decode, process_headtail, select_topk, create_word_mask, mask_logits, get_logits_embedding
from transformers import DynamicCache
# from colorama import Fore, Style


def soft_forward(model, head_tokens, prompt_logits, 
                   tail_tokens, target_tokens=None):
    embedding_layer = model.get_input_embeddings()  # Usually model.embeddings or model.get_input_embeddings()
    
    # Step 2: Hard-select embeddings for head and tail
    head_embeddings =  embedding_layer(head_tokens)  # Shape: (batch_size, head_length, embedding_dim)
    tail_embeddings = embedding_layer(tail_tokens)  # Shape: (batch_size, tail_length, embedding_dim)

    # Step 3: Soft-select embeddings for prompt
    prompt_probs = torch.softmax(prompt_logits, dim=-1)  # Shape: (batch_size, prompt_length, vocab_size)
    vocab_embeddings = embedding_layer.weight  # Embedding matrix (vocab_size, embedding_dim)
    prompt_embeddings = torch.matmul(prompt_probs, vocab_embeddings)  # Shape: (batch_size, prompt_length, embedding_dim)

    total = [head_embeddings, prompt_embeddings, tail_embeddings]

    start = head_tokens.shape[1] - 1
    end = start + prompt_logits.shape[1]

    # Step 4: add target embeddings if provided
    if target_tokens is not None:
        target_embeddings = embedding_layer(target_tokens)  # Shape: (batch_size, target_length, embedding_dim)
        total.append(target_embeddings)
        start = -1 - target_tokens.shape[1]
        end = -1

    sequence_embeddings = torch.cat(total, dim=1)

    # Step 5: Forward pass through the model
    logits = model(inputs_embeds=sequence_embeddings).logits
    # return the prompt logits
    specific_logits = logits[:, start:end, :]

    return specific_logits


def fluency_loss(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def bleu_loss(decoder_outputs, target_idx, ngram_list, pad=0, weight_list=None):
    batch_size, output_len, _ = decoder_outputs.size()
    _, tgt_len = target_idx.size()
    if type(ngram_list) == int:
        ngram_list = [ngram_list]
    if ngram_list[0] <= 0:
        ngram_list[0] = output_len
    if weight_list is None:
        weight_list = [1. / len(ngram_list)] * len(ngram_list)
    decoder_outputs = torch.log_softmax(decoder_outputs,dim=-1)
    decoder_outputs = torch.relu(decoder_outputs + 20) - 20
    # target_idx expand to batch size * tgt len
    target_idx = target_idx.expand(batch_size, -1)
    index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
    cost_nll = decoder_outputs.gather(dim=2, index=index)
    cost_nll = cost_nll.unsqueeze(1)
    out = cost_nll
    sum_gram = 0. 

    zero = torch.tensor(0.0).to(decoder_outputs.device)
    target_expand = target_idx.view(batch_size,1,1,-1).expand(-1,-1,output_len,-1)
    out = torch.where(target_expand==pad, zero, out)

    for cnt, ngram in enumerate(ngram_list):
        if ngram > output_len:
            continue
        eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).to(decoder_outputs.device)
        term = nn.functional.conv2d(out, eye_filter)/ngram
        if ngram < decoder_outputs.size()[1]:
            term = term.squeeze(1)
            gum_tmp = F.gumbel_softmax(term, tau=1, dim=1)
            term = term.mul(gum_tmp).sum(1).mean(1)
        else:
            while len(term.shape) > 1:
                assert term.shape[-1] == 1, str(term.shape)
                term = term.sum(-1)
        sum_gram += weight_list[cnt] * term

    loss = - sum_gram
    return loss






def add_static_noise(prompt_logits, iter, iter_steps, noise_stds):

    def get_noise_std(iteration):
        for i, step in enumerate(iter_steps):
            if iteration < step:
                return noise_stds[i - 1] if i > 0 else noise_stds[0]
        return noise_stds[-1]
            
    noise_std = get_noise_std(iter)

    noise = torch.normal(mean=0.0, std=noise_std, size=prompt_logits.size(),
                        device=prompt_logits.device, requires_grad=False)
    
    return prompt_logits + noise


def rank_products(text, product_names):
    position_dict = {}
    for name in product_names:
        position = text.find(name)
        if position != -1:
            position_dict[name] = position
        else:
            position_dict[name] = float('inf')

    # Sort products by position
    sorted_products = sorted(position_dict, key=position_dict.get)

    ranks = {}
    for i, prod in enumerate(sorted_products):
        if position_dict[prod] != float('inf'):
            ranks[prod] = i + 1
        else:
            ranks[prod] = len(sorted_products) + 1

    return ranks


# def print_iteration_metrics(iteration, total_loss, fluency_loss, n_gram_loss, target_loss):
#     log_message = (
#         f"{Fore.GREEN}Iteration {iteration+1}: {Style.RESET_ALL}"
#         f"{Fore.YELLOW}Total Loss: {total_loss:.4f} {Style.RESET_ALL}"
#         f"{Fore.BLUE}Fluency Loss: {fluency_loss:.4f} {Style.RESET_ALL}"
#         f"{Fore.CYAN}N-gram Loss: {n_gram_loss:.4f} {Style.RESET_ALL}"
#         f"{Fore.MAGENTA}Target Loss: {target_loss:.4f} {Style.RESET_ALL}"
#     )
#     print(f"{log_message}", flush=True)

def topk_decode(model, tokenizer, y_logits, topk, temperature, head_tokens):
    # empty cache
    prompt_length = y_logits.shape[1]

    embedding_layer = model.get_input_embeddings()
    
    head_embed = embedding_layer(head_tokens)

    # to get past key values for efficient decoding
    past_embed = head_embed[:, :-1, :]
    past = model(inputs_embeds=past_embed, use_cache=True).past_key_values

    input_embeds = head_embed[:, -1:, :]

    for i in range(prompt_length):
        output = model(inputs_embeds=input_embeds, past_key_values=past, use_cache=True)
        past = output.past_key_values
        last_logits = output.logits[:, -1:, :]
        topk_mask = select_topk(last_logits, topk)

        total_topk_mask = topk_mask if i == 0 else torch.cat([total_topk_mask, topk_mask], dim=1)

        if i < prompt_length - 1:
            current_y_logits = mask_logits(y_logits[:, i:i+1, :], topk_mask) / temperature
            input_embeds = torch.matmul(F.softmax(current_y_logits, dim=-1).to(embedding_layer.weight.dtype),
                                         embedding_layer.weight)

    complete_masked_logits = mask_logits(y_logits, total_topk_mask)

    decoded_tokens, decoded_text = greedy_decode(complete_masked_logits, tokenizer)

    return decoded_tokens, decoded_text, complete_masked_logits


def gcg_decode(model, tokenizer, head_tokens, y_logits, topk, tail_tokens, target_tokens):
    # apply one step of greedy coordinate gradient focusing only on target loss
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight.data

    head_embeddings = embedding_layer(head_tokens)
    tail_embeddings = embedding_layer(tail_tokens)
    target_embeddings = embedding_layer(target_tokens)

    prompt_probs = torch.softmax(y_logits, dim=-1)
    prompt_embeddings = torch.matmul(prompt_probs, embedding_matrix)
    input_embeddings = torch.cat([head_embeddings, prompt_embeddings, tail_embeddings], dim=1)
    input_embeddings.retain_grad()

    total_embeddings = torch.cat([input_embeddings, target_embeddings], dim=1)
    
    logits = model(inputs_embeds=total_embeddings).logits

    loss_fn = nn.CrossEntropyLoss()

    loss = []

    for i in range(input_embeddings.shape[0]):
        loss.append(loss_fn(logits[i, input_embeddings.shape[1]-1:-1, :], target_tokens[i]))

    loss = torch.stack(loss).sum()

    (-loss).backward()
    gradients = input_embeddings.grad

    start = head_tokens.shape[1] - 1
    end = start + y_logits.shape[1]

    dot_prod = torch.matmul(gradients, embedding_matrix.T)[:, start:end, :]

    topk_mask = select_topk(y_logits, topk)

    # mask the dot_prod
    dot_prod = mask_logits(dot_prod, topk_mask)

    decoded_tokens, decoded_text = greedy_decode(dot_prod, tokenizer)

    return decoded_tokens, decoded_text



def log_result(model, tokenizer, head_tokens, logits, tail_tokens, target_tokens,
               iter, product_list, target_product, result_table, logger, topk, temperature):
    
    product_names = [product['Name'] for product in product_list]

    # prompt_tokens, decoded_prompt, _ = topk_decode(model, tokenizer, logits, topk, temperature, head_tokens)
    # use greedy decode
    prompt_tokens, decoded_prompt = greedy_decode(logits, tokenizer)

    # use gcg decode
    # with torch.autocast(device_type=model.device.type, dtype=torch.float16):
    #     prompt_tokens, decoded_prompt = gcg_decode(model, tokenizer, head_tokens, logits, topk, tail_tokens, target_tokens)

    # Concatenate head prompt, decoded text, and tail tokens
    complete_prompt = torch.cat([head_tokens, prompt_tokens, tail_tokens], dim=1)

    # Generate result from model using the complete prompt
    batch_result = model.generate(complete_prompt, model.generation_config, 
                                  max_new_tokens=500, 
                                #   do_sample=True, temperature=0.7, top_k=topk, # NOT sure if we need this 2 parameters. used in cold-attack but not in product ranking
                                  attention_mask=torch.ones_like(complete_prompt), 
                                  pad_token_id=tokenizer.eos_token_id)

    # index batch_result to avoid the input prompt
    batch_result = batch_result[:, complete_prompt.shape[1]:]

    generated_texts = tokenizer.batch_decode(batch_result, skip_special_tokens=True)

    product_ranks = []
    current_table = wandb.Table(columns=result_table.columns, data=result_table.data)
    
    # Log the complete prompt and the generated result in a JSON file
    for i in range(len(decoded_prompt)):
        current_rank = rank_products(generated_texts[i], product_names)[target_product]
        product_ranks.append(current_rank)
        highlighted_attack_prompt = f'<span style="color:red;">{decoded_prompt[i]}</span>'
        current_complete_prompt = tokenizer.decode(complete_prompt[i], skip_special_tokens=True)

        log_entry = {
            "attack_prompt": highlighted_attack_prompt,
            "complete_prompt": current_complete_prompt.replace(decoded_prompt[i], highlighted_attack_prompt),
            "generated_result": generated_texts[i],
            'product_rank': current_rank
        }
        
        current_table.add_data(iter,
                            log_entry["attack_prompt"],
                            log_entry["complete_prompt"],
                            log_entry["generated_result"],
                            log_entry["product_rank"])

    logger.log({f"eval/target:{target_product}": current_table}, step=iter)

    # print(f"{Fore.RED}Evaluation results have been saved to {output_file} at iteration {iter+1}{Style.RESET_ALL}")
    # print(f"{Fore.RED}Target Product: {target_product} Rank: {min(product_ranks)} (11 means not found in the generated text){Style.RESET_ALL}")
    
    logger.log({"eval/product_rank": min(product_ranks),
                "eval/batch_idx": product_ranks.index(min(product_ranks))}, step=iter)
    
    return current_table, min(product_ranks)



def soft_log_result(model, tokenizer, head_tokens, logits, tail_tokens, 
               iter, product_list, target_product, result_table, logger, topk, temperature):
    torch.cuda.empty_cache()
    product_names = [product['Name'] for product in product_list]

    # _, _, masked_logits = topk_decode(model, tokenizer, logits, topk, temperature, head_tokens)

    embedding_layer = model.get_input_embeddings()  # Usually model.embeddings or model.get_input_embeddings()
    
    # Step 2: Hard-select embeddings for head and tail
    head_embeddings =  embedding_layer(head_tokens)  # Shape: (batch_size, head_length, embedding_dim)
    tail_embeddings = embedding_layer(tail_tokens)  # Shape: (batch_size, tail_length, embedding_dim)
    # Step 3: Soft-select embeddings for prompt
    # prompt_probs = torch.softmax(masked_logits, dim=-1)  # Shape: (batch_size, prompt_length, vocab_size)
    # vocab_embeddings = embedding_layer.weight  # Embedding matrix (vocab_size, embedding_dim)

    # prompt_embeddings = torch.matmul(prompt_probs, vocab_embeddings)  # Shape: (batch_size, prompt_length, embedding_dim)

    # hard select prompt embeddings
    prompt_embeddings = embedding_layer(logits.argmax(dim=-1))

    total_embeddings = torch.cat([head_embeddings, prompt_embeddings, tail_embeddings], dim=1)

    result = []
    past_embed = total_embeddings[:, :-1, :]
    past = model(inputs_embeds=past_embed, use_cache=True).past_key_values
    past = DynamicCache.from_legacy_cache(past)

    input_embeds = total_embeddings[:, -1:, :]

    for i in range(800-total_embeddings.shape[1]):
        output = model(inputs_embeds=input_embeds, past_key_values=past, use_cache=True)
        past = DynamicCache.from_legacy_cache(output.past_key_values)
        last_logits = output.logits[:, -1:, :]

        token = torch.argmax(last_logits, dim=-1)
        result.append(token)
        input_embeds = model.get_input_embeddings()(token)

    batch_result = torch.cat(result, dim=1)
    generated_texts = tokenizer.batch_decode(batch_result, skip_special_tokens=True)

    product_ranks = []
    current_table = wandb.Table(columns=result_table.columns, data=result_table.data)
    
    # Log the complete prompt and the generated result in a JSON file
    for i in range(total_embeddings.shape[0]):
        current_rank = rank_products(generated_texts[i], product_names)[target_product]
        product_ranks.append(current_rank)
        # highlighted_attack_prompt = f'<span style="color:red;">{decoded_prompt[i]}</span>'
        # decode head and tail separately
        current_head = tokenizer.decode(head_tokens[i], skip_special_tokens=True)
        current_tail = tokenizer.decode(tail_tokens[i], skip_special_tokens=True)

        log_entry = {
            "attack_prompt": 'soft',
            "complete_prompt": current_head + '<span style="color:red;">soft</span>' + current_tail,
            "generated_result": generated_texts[i],
            'product_rank': current_rank
        }
        
        current_table.add_data(iter,
                            log_entry["attack_prompt"],
                            log_entry["complete_prompt"],
                            log_entry["generated_result"],
                            log_entry["product_rank"])

    logger.log({f"eval/target:{target_product}": current_table}, step=iter)

    # print(f"{Fore.RED}Evaluation results have been saved to {output_file} at iteration {iter+1}{Style.RESET_ALL}")
    # print(f"{Fore.RED}Target Product: {target_product} Rank: {min(product_ranks)} (11 means not found in the generated text){Style.RESET_ALL}")
    
    logger.log({"eval/product_rank": min(product_ranks),
                "eval/batch_idx": product_ranks.index(min(product_ranks))}, step=iter)
    
    return current_table, min(product_ranks)


def attack_control(model, tokenizer, system_prompt, user_msg,
                   prompt_logits, target_tokens, extra_word_tokens, 
                   product_list, target_product, logger, table,
                   original_embedding=None, **kwargs):

    mode = kwargs['mode'] # suffix or paraphrase
    device = model.device
    epsilon = nn.Parameter(torch.zeros_like(prompt_logits))
    # kaiming initialize  
    # nn.init.kaiming_normal_(epsilon)
    optimizer = torch.optim.Adam([epsilon], lr=kwargs['lr'])
    batch_size = prompt_logits.size(0)

    extra_mask = create_word_mask(extra_word_tokens, prompt_logits)
    
    topk_mask = None
    current_rank = None

    with tqdm(total=kwargs['num_iter'], desc="Training", unit="iter") as pbar:
        for iter in range(kwargs['num_iter']):  
            if kwargs['random_order']:
                # shuffle the product list
                random.shuffle(product_list)

            # train with target product at last
            head_tokens, tail_tokens = process_headtail(tokenizer, system_prompt, product_list, user_msg, 
                                                        target_product, batch_size, mode, device, last=True)
            # add learnable noise to prompt logits
            y_logits = prompt_logits + epsilon

            # fluency loss
            if topk_mask is None:
                fluency_soft_logits = (y_logits.detach() / kwargs['temperature'] - y_logits).detach() + y_logits
            else:
                fluency_soft_logits = mask_logits(y_logits, topk_mask, extra_mask) / kwargs['temperature']

            with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")):
                perturbed_y_logits = soft_forward(model, head_tokens, fluency_soft_logits, tail_tokens).detach()
            
            topk_mask = select_topk(perturbed_y_logits, kwargs['topk'])

            perturbed_y_logits = mask_logits(perturbed_y_logits, topk_mask, extra_mask)

            flu_loss = fluency_loss(perturbed_y_logits, y_logits)
            
            # target loss
            # soft_logits = y_logits / kwargs['temperature']  
            soft_logits = (y_logits.detach() / kwargs['temperature'] - y_logits).detach() + y_logits
            with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")):
                target_logits = soft_forward(model, head_tokens, soft_logits, tail_tokens, target_tokens)
                
            target_loss = nn.CrossEntropyLoss(reduction='none')(
                target_logits.reshape(-1, target_logits.size(-1)), 
                target_tokens.view(-1))
            target_loss = target_loss.view(batch_size, -1).mean(dim=-1)

            # ngram bleu loss
            if mode == 'suffix':
                n_gram_loss = bleu_loss(y_logits, extra_word_tokens, ngram_list=[1])
                total_loss = kwargs['fluency'] * flu_loss - kwargs['ngram'] * n_gram_loss + kwargs['target'] * target_loss
                similarity_score = None

            elif mode == 'paraphrase':
                mask_y_logits = mask_logits(y_logits, topk_mask, extra_mask)
                n_gram_loss = bleu_loss(mask_y_logits, extra_word_tokens, ngram_list=[1, 2, 3])

                logits_embedding = get_logits_embedding(model, y_logits, kwargs['temperature'])

                similarity_score = F.cosine_similarity(original_embedding, logits_embedding)
                total_loss = kwargs['fluency'] * flu_loss + kwargs['ngram'] * n_gram_loss + kwargs['target'] * target_loss - kwargs['similarity'] * similarity_score

            
            # total loss weighted sum
            
            total_loss = total_loss.mean()
            
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.2f}",
                "Target": f"{target_loss.mean().item():.2f}",
                "Fluency": f"{flu_loss.mean().item():.2f}",
                "N-gram": f"{n_gram_loss.mean().item():.2f}",
                "Similarity": f"{similarity_score.mean().item():.2f}" if similarity_score is not None else "N/A",
                "Rank": f"{current_rank}"
            })
            

            logger.log({"train/loss": total_loss.item(), 
                        "train/fluency_loss": flu_loss.mean().item(), 
                        "train/ngram_loss": n_gram_loss.mean().item(), 
                        "train/target_loss": target_loss.mean().item()},
                        step=iter)
            
            if similarity_score:
                logger.log({"train/similarity": similarity_score.mean().item()}, step=iter)


            # evaluate and log into a jsonl file
            if iter == 0 or (iter+1) % kwargs['test_iter'] == 0 or iter == kwargs['num_iter'] - 1:
                # with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")), torch.no_grad():
                #     table, rank = soft_log_result(model, tokenizer, head_tokens, y_logits, tail_tokens, 
                #             iter, product_list, target_product, table, logger, kwargs['topk'], kwargs['temperature'])
                    
                # inference at random places
                if kwargs['random_inference']:
                    head_tokens, tail_tokens = process_headtail(tokenizer, system_prompt, product_list, user_msg,
                                                                    target_product, batch_size, device, last=False)

                table, rank = log_result(model, tokenizer, head_tokens, y_logits, tail_tokens, target_tokens,
                                iter, product_list, target_product, table, logger, kwargs['topk'], kwargs['temperature'])
                
                current_rank = rank
                tqdm.write(
                    f"Evaluation completed at iteration {iter + 1} Rank: {current_rank}"
                )
            pbar.update(1)
            
            # add static noise and do not add either static or learnable noise at the last iteration
            if iter < kwargs['num_iter'] - 1:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                prompt_logits = add_static_noise(prompt_logits, iter, kwargs['iter_steps'], kwargs['noise_stds'])
        

    


    
