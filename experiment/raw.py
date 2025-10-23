import torch, random, transformers
import numpy as np

import argparse

from experiment.main import ASSSISTANT_PROMPT, MODEL_PATH_DICT, SYSTEM_PROMPT
from experiment.get import get_model, get_product_list, get_user_query
from experiment.attack import rank_products

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ASSSISTANT_PROMPT = "You are a recommendation AI designed to assist users in selecting the best products based on their preferences and needs. Provide a numbered list of product recommendations ranked according to the user's request."

MODEL_PATH_DICT = {'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                #    'llama-2-7b': "meta-llama/Llama-2-7b-chat-hf", 
                   'vicuna-7b': "lmsys/vicuna-7b-v1.5",
                   'vicuna-13b': "lmsys/vicuna-13b-v1.3",
                   'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
                   'deepseek-7b': 'deepseek-ai/deepseek-llm-7b-chat',
                   'llama-3-13b': 'elinas/Llama-3-13B-Instruct',}
                #    'Qwen-2.5-14B': 'Qwen/Qwen2.5-14B-Instruct'}


SYSTEM_PROMPT = {'llama': {'head': f'[INST] <<SYS>>\n{ASSSISTANT_PROMPT}\n<<SYS>>\n\n', 
                            'tail': ' [/INST]'},
                'vicuna': {'head': f'{ASSSISTANT_PROMPT}\n\nUser:',
                           'tail': '\n\nAssistant: '},
                'mistral': {'head': f'<s>[INST] {ASSSISTANT_PROMPT}\n\n',
                            'tail': ' [/INST]'},
                'deepseek': {'head': f'{ASSSISTANT_PROMPT}\n\nUser:',
                             'tail': '\n\nAssistant: '},
                'Qwen':    {'head': f'<|im_start|>system\n{ASSSISTANT_PROMPT}<|im_end|>\n<|im_start|>user\n',
                             'tail': '<|im_end|>\n<|im_start|>assistant\n'}}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = 'json'
    catalogs = ['books', 'cameras', 'coffee_machines']
    indices = [i for i in range(1,11)]

    results = []

    for model_name, model_path in MODEL_PATH_DICT.items():
        print(f'Processing {model_name}...')
        model, tokenizer = get_model(model_path, '16', device)

        model_ranks = []
        for catalog in catalogs:
            user_query = get_user_query(catalog)

            catalog_ranks = []
            for idx in indices:
                product_list, target_product, target_product_natural, target_str = get_product_list(catalog, idx, dataset)
                full_prompt = SYSTEM_PROMPT[model_name.split('-')[0]]['head'] + user_query + '\n'.join([p['Natural'] for p in product_list]) + SYSTEM_PROMPT[model_name.split('-')[0]]['tail'] 

                encoded_full_prompt = tokenizer(full_prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = model.generate(encoded_full_prompt, 
                                            model.generation_config, 
                                            max_new_tokens=500, 
                                            attention_mask=torch.ones_like(encoded_full_prompt), 
                                            pad_token_id=tokenizer.eos_token_id)
                    
                    for seq in output:
                        decoded_seq = tokenizer.decode(seq, skip_special_tokens=True)
                        
                        rank = rank_products(decoded_seq, [p['Name'] for p in product_list])[target_product]

                catalog_ranks.append(rank)

            avg_catalog_rank = np.mean(catalog_ranks)
            model_ranks.append(avg_catalog_rank)

        model_avg_rank = np.mean(model_ranks)
        results.append({'Model': model_name,
                        'Average Rank': f'{round(model_avg_rank, 2)}'})

        df_results = pd.DataFrame(results)
        print(tabulate(df_results, headers='keys', tablefmt='grid'))

        save_path = "raw.csv"
        df_results.to_csv(save_path, index=False)
        print(f"✅ Saved to {save_path}")

            
if __name__ == '__main__':
    main()

