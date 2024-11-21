import torch, json, random
import torch.nn as nn
import torch.nn.functional as F
from experiment.process import greedy_decode, proces_headtail, select_topk, create_bad_words_mask, mask_logits
from colorama import Fore, Style


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

    # Step 4: Hard-select embeddings for target (if provided)
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


def add_static_noise(prompt_logits, iter, device):
    iter_steps = [50, 200, 500, 1500]
    noise_stds = [0.1, 0.05, 0.01, 0.001]

    def get_noise_std(iter, iter_steps, noise_stds):
        for i, step in enumerate(iter_steps):
            if iter < step:
                return noise_stds[i]
            
    noise_std = get_noise_std(iter, iter_steps, noise_stds)

    noise = torch.normal(mean=0, std=noise_std, size=prompt_logits.size(),
                        device=device, requires_grad=False)
    
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


def print_iteration_metrics(iteration, total_loss, fluency_loss, n_gram_loss, target_loss):
    log_message = (
        f"{Fore.GREEN}Iteration {iteration+1}: {Style.RESET_ALL}"
        f"{Fore.YELLOW}Total Loss: {total_loss:.4f} {Style.RESET_ALL}"
        f"{Fore.BLUE}Fluency Loss: {fluency_loss:.4f} {Style.RESET_ALL}"
        f"{Fore.CYAN}N-gram Loss: {n_gram_loss:.4f} {Style.RESET_ALL}"
        f"{Fore.MAGENTA}Target Loss: {target_loss:.4f} {Style.RESET_ALL}"
    )
    print(f"{log_message}", flush=True)


def log_result(model, tokenizer, head_tokens, logits, tail_tokens, 
               iter, product_list, target_product, output_file):
    
    product_names = [product['Name'] for product in product_list]

    prompt_tokens, decoded_prompt = greedy_decode(logits, tokenizer)

    # Concatenate head prompt, decoded text, and tail tokens
    complete_prompt = torch.cat([head_tokens, prompt_tokens, tail_tokens], dim=1)

    # Generate result from model using the complete prompt
    batch_result = model.generate(complete_prompt, model.generation_config, max_new_tokens=800, 
                                  attention_mask=torch.ones_like(complete_prompt))

    # index batch_result to avoid the input prompt

    batch_result = batch_result[:, complete_prompt.shape[1]:]

    generated_texts = tokenizer.batch_decode(batch_result, skip_special_tokens=True)

    product_ranks = []

    # Log the complete prompt and the generated result in a JSON file
    with open(output_file, "a") as f:
        f.write(f"Evaluating at Iteration {iter}\n")
        for i in range(len(decoded_prompt)):
            current_ranks = rank_products(generated_texts[i], product_names)[target_product]
            product_ranks.append(current_ranks)
            log_entry = {
                "attack_prompt": decoded_prompt[i],
                "complete_prompt": tokenizer.decode(complete_prompt[i], skip_special_tokens=True),
                "generated_result": generated_texts[i],
                'product_rank': current_ranks
            }
            f.write(json.dumps(log_entry, indent=4) + "\n")

    print(f"{Fore.RED}Evaluation results have been saved to {output_file} at iteration {iter+1}{Style.RESET_ALL}")

    # return the min product rank and the batch index of that prompt
    return min(product_ranks), product_ranks.index(min(product_ranks))
    


def attack_control(model, tokenizer, system_prompt, user_msg,
                   prompt_logits, target_tokens, bad_words_tokens, 
                   product_list, target_product, logger, **kwargs):

    device = model.device
    epsilon = nn.Parameter(torch.zeros_like(prompt_logits))
    print('epsilon dtype:', epsilon.dtype) 
    # kaiming initialize  
    # nn.init.kaiming_normal_(epsilon)
    optimizer = torch.optim.Adam([epsilon], lr=kwargs['lr'])
    batch_size = prompt_logits.size(0)

    topk_mask = None
    bad_words_mask = create_bad_words_mask(bad_words_tokens, prompt_logits)
    target_tokens = target_tokens.long()

    for iter in range(kwargs['num_iter']):  
        if kwargs['random_order']:
            # shuffle the product list
            random.shuffle(product_list)

        head_tokens, tail_tokens = proces_headtail(tokenizer, system_prompt, product_list, user_msg, target_product, batch_size, device)

        head_tokens = head_tokens.long()
        tail_tokens = tail_tokens.long()
        # add learnable noise to prompt logits
        y_logits = prompt_logits + epsilon

        # ngram bleu loss
        with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")):
            n_gram_loss = bleu_loss(y_logits, bad_words_tokens, ngram_list=[1])

        # fluency loss
        if topk_mask is None:
            fluency_soft_logits = (y_logits.detach() / 0.001 - y_logits).detach() + y_logits
        else:
            fluency_soft_logits = mask_logits(y_logits, topk_mask, bad_words_mask) / 0.001

        with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")):
            perturbed_y_logits = soft_forward(model, head_tokens, fluency_soft_logits, tail_tokens).detach()
        
        topk_mask = select_topk(perturbed_y_logits, kwargs['topk'])

        perturbed_y_logits = mask_logits(perturbed_y_logits, topk_mask, bad_words_mask)

        flu_loss = fluency_loss(perturbed_y_logits, y_logits)
        
        # target loss
        soft_logits = (y_logits.detach() / 0.001 - y_logits).detach() + y_logits
        with torch.autocast(device_type=device.type, dtype=eval(f"torch.float{kwargs['precision']}")):
            target_logits = soft_forward(model, head_tokens, soft_logits, tail_tokens, target_tokens)
            
        target_loss = nn.CrossEntropyLoss(reduction='none')(
            target_logits.reshape(-1, target_logits.size(-1)), 
            target_tokens.view(-1))
        target_loss = target_loss.view(batch_size, -1).mean(dim=-1)
        
        # total loss weighted sum
        loss_weights = kwargs['loss_weights']
        total_loss = loss_weights['fluency'] * flu_loss - loss_weights['ngram'] * n_gram_loss + loss_weights['target'] * target_loss
        total_loss = total_loss.mean()
        
        # log the loss
        print_iteration_metrics(iter, total_loss.item(), flu_loss.mean().item(), n_gram_loss.mean().item(), target_loss.mean().item())
        logger.log({"train/loss": total_loss.item(), 
                    "train/fluency_loss": flu_loss.mean().item(), 
                    "train/ngram_loss": n_gram_loss.mean().item(), 
                    "train/target_loss": target_loss.mean().item()}, 
                    step=iter)


        # evaluate and log into a jsonl file
        if iter % kwargs['test_iter'] == 0 or iter == kwargs['num_iter'] - 1:
            product_rank, batch_idx = log_result(model, tokenizer, head_tokens, y_logits, tail_tokens, 
                       iter, product_list, target_product, kwargs['result_file'])
            logger.log({"eval/product_ranks": product_rank,
                        "eval/batch_idx": batch_idx}, step=iter)
            
        
        # add static noise and do not add either static or learnable noise at the last iteration
        if iter < kwargs['num_iter'] - 1:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            prompt_logits = add_static_noise(prompt_logits, iter, device)
        

    


    
