import torch, os, wandb, json, yaml
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from experiment.get import get_user_query, get_model, get_product_list
from experiment.process import process_bad_words, greedy_decode, init_prompt, process_target
from experiment.constant import MODEL_PATH_DICT, SYSTEM_PROMPT, BAD_WORDS
from experiment.attack import attack_control


ENTITY = 'fanyieee-university-of-southern-california'
PROJECT = 'seo'
# def get_args():
#     argparser = argparse.ArgumentParser(description="Product Rank Optimization")
#     argparser.add_argument("--model", type=str, default="llama-3.1-8b", choices=["llama-3.1-8b", "llama-2-7b"], help="The model to use.")
#     argparser.add_argument("--batch_size", type=int, default=4, help="The batch size.")
#     argparser.add_argument("--length", type=int, default=50, help="The length of the generated text.")
#     argparser.add_argument("--temperature", type=float, default=1.0, help="The temperature of the sampling.")

#     argparser.add_argument("--lr", type=float, default=0.1, help="The learning rate.")
#     argparser.add_argument("--topk", type=int, default=10, help="The top-k value.")
#     argparser.add_argument("--num_iter", type=int, default=2000, help="The number of iterations.")
#     argparser.add_argument("--test_iter", type=int, default=200, help="The number of test iterations.")
#     argparser.add_argument("--precision", type=int, default=16, help="The precision of the model.")
#     argparser.add_argument("--loss_weights", type=json.loads, 
#                            default='{"fluency": 1.0, "ngram": 100.0, "target": 100.0}', 
#                            help="Loss weights as a JSON string. Example: '{\"fluency\": 1.0, \"ngram\": 1.0, \"target\": 1.0}'")


#     argparser.add_argument("--result_dir", type=str, default="result", help="The directory to save the results.")
#     argparser.add_argument("--catalog", type=str, default="coffee_machines", choices=["election_articles","coffee_machines", "books", "cameras"], help="The product catalog to use.")
#     argparser.add_argument("--random_order", action="store_true", help="Whether to shuffle the product list in each iteration.")
#     argparser.add_argument("--target_product_idx", type=int, default=1, help="The index of the target product in the product list.")
#     argparser.add_argument("--mode", type=str, default="self", choices=["self", "transfer"], help="Mode of optimization.")
#     argparser.add_argument("--user_msg_type", type=str, default="default", choices=["default", "custom"], help="User message type.")
#     #argparser.add_argument("--save_state", action="store_true", help="Whether to save the state of the optimization procedure. If interrupted, the experiment can be resumed.")
#     return argparser.parse_args()



def log_init_prompt(result_dir, decoded_sentences):
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/eval.jsonl'

    n = 1
    while os.path.exists(output_file):
        output_file = f'{result_dir}/eval-v{n}.jsonl'
        n += 1

    with open(output_file, "a") as f:
        f.write('Initial Prompts:\n')
        for sentence in decoded_sentences:
            f.write(json.dumps({"attack_prompt": sentence}, indent=4) + "\n")
        
    return output_file

def main():
    wandb_logger = wandb.init(entity=ENTITY, project=PROJECT)
    wandb_table = wandb.Table(columns=["iter", "attack_prompt", "complete_prompt", "generated_result", "product_rank"])
    hparams = wandb.config

    # log all hparams
    wandb_logger.name = 'llm-seo'

    user_msg = get_user_query(hparams.user_msg_type, hparams.catalog)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model(MODEL_PATH_DICT[hparams.model], hparams.precision, device)

    product_list, target_product, target_str = get_product_list(hparams.catalog, hparams.target_product_idx)
    print("\nTARGET STR:", target_str)

    prompt_logits = init_prompt(model=model, 
                                tokenizer=tokenizer, 
                                product_list=product_list, 
                                target_product_idx=hparams.target_product_idx-1, 
                                temperature=hparams.temperature, 
                                prompt_length=hparams.length, 
                                batch_size=hparams.batch_size, 
                                device=device)
    
    target_tokens = process_target(tokenizer=tokenizer, 
                                   target_str=target_str, 
                                   batch_size=hparams.batch_size, 
                                   device=device)
    
    # Print initial prompt logits
    _, decoded_init_prompt = greedy_decode(logits=prompt_logits, tokenizer=tokenizer)
    output_file = log_init_prompt(hparams.result_dir, decoded_init_prompt)

    wandb_logger.config.update({'output_file': output_file})

    # Process the bad words tokens
    bad_words_tokens = process_bad_words(bad_words=BAD_WORDS, 
                                         tokenizer=tokenizer, 
                                         device=device)

    # Attack control
    
    attack_control(model=model, 
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT[hparams.model.split("-")[0]],
                user_msg=user_msg,
                prompt_logits=prompt_logits,  
                target_tokens=target_tokens, 
                bad_words_tokens=bad_words_tokens, 
                product_list=product_list,
                target_product=target_product,
                logger=wandb_logger,
                result_file=output_file,
                table=wandb_table,
                num_iter=hparams.num_iter, 
                test_iter=hparams.test_iter,
                topk=hparams.topk, 
                lr=hparams.lr, 
                precision=hparams.precision,
                loss_weights=hparams.loss_weights,
                random_order=hparams.random_order)
    
    wandb_logger.finish()



if __name__ == "__main__":
    with open(f'experiment/config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
    wandb.agent(sweep_id, function=main, entity=ENTITY, project=PROJECT)

