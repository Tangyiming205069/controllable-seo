from experiment.get import get_model, get_product_list, get_user_query
from experiment.attack import rank_products
from experiment.main import MODEL_PATH_DICT, SYSTEM_PROMPT, ASSSISTANT_PROMPT, ORDERING_PROMPT
import torch, os, random
import pandas as pd
import statistics
from tabulate import tabulate
from openai import OpenAI
client = OpenAI()

openai_key = ''


def compose_transfer_prompt(product_list, attack_prompt, target_product, user_msg):
    # Compose the transfer prompt
    product_names = [product['Name'] for product in product_list]
    target_product_idx = product_names.index(target_product)

    result = user_msg + "\n\nProducts:\n"

    for i, product in enumerate(product_list):
        if i < target_product_idx:
            result += product['Natural'] + "\n"
        elif i == target_product_idx:
            result += product['Natural'] + attack_prompt + "\n"
        else:
            result += product['Natural'] + "\n"
    result = result.rstrip('\n')
    return result


def transfer_evaluate(dataset, catalog, base_model, transfer_models, result_dir, output_dir, indices):
    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    usr_msg = get_user_query(catalog)

    for transfer_model in transfer_models:
        model, tokenizer = get_model(MODEL_PATH_DICT[transfer_model], 16, device)
        transfer_ranks = []

        for index in indices:
            file_path = f"{result_dir}/{base_model}/{dataset}/{catalog}/{index}/random_inference=True.csv"

            if not os.path.exists(file_path):
                print(f"❌ {file_path} does not exist.")
                continue
            
            product_list, target_product, _, _ = get_product_list(catalog, index, dataset)

            df = pd.read_csv(file_path)s
            

            # take the last 5 rows
            df = df.tail(5) # last batch
            ranks = df['product_rank'].tolist()
            min_rank = min(ranks)
            row = df[df['product_rank'] == min_rank]

            attack_prompt = row['attack_prompt'].values[0]
            # complete_prompt = row['complete_prompt'].values[0]
            # body_prompt = complete_prompt.lstrip(SYSTEM_PROMPT[base_model.split('-')[0]]['head']).rstrip(SYSTEM_PROMPT[base_model.split('-')[0]]['tail'])
            # body_prompt = body_prompt.replace('<span style="color:red;">', '').replace('</span>', '')
            
            attack_prompt = attack_prompt.replace('<span style="color:red;">', '').replace('</span>', '')

            target_product = product_list[index - 1]['Name']

            random.shuffle(product_list)
            transfer_user_prompt = compose_transfer_prompt(product_list, attack_prompt, target_product, usr_msg)

            transfer_prompt = SYSTEM_PROMPT[transfer_model.split('-')[0]]['head'] + transfer_user_prompt + SYSTEM_PROMPT[transfer_model.split('-')[0]]['tail']
            import pdb; pdb.set_trace()
            input_ids = tokenizer(transfer_prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                output = model.generate(input_ids=input_ids['input_ids'], 
                                        max_new_tokens=500, 
                                        attention_mask=torch.ones_like(input_ids.input_ids), 
                                        pad_token_id=tokenizer.eos_token_id)

            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

            
            product_names = [product['Name'] for product in product_list]
            rank = rank_products(decoded_output, product_names)[target_product]

            transfer_ranks.append(rank)

        # Save the ranks to a CSV file
        avg_rank = sum(transfer_ranks) / len(transfer_ranks)
        std_rank = statistics.stdev(transfer_ranks)

        results.append({
            'base_model': base_model,
            'transfer_model': transfer_model,
            'dataset': dataset,
            'catalog': catalog,
            "Average Rank": f'{round(avg_rank, 2)}±{round(std_rank, 2)}', 
        })

    df_results = pd.DataFrame(results)
    print(tabulate(df_results, headers='keys', tablefmt='grid'))

    output_dir = f"{output_dir}/{dataset}/{base_model}"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/{catalog}.csv"
    df_results.to_csv(save_path, index=False)
    print(f"✅ Saved to {save_path}")


def transfer_gpt(dataset, catalog, base_model, transfer_models, result_dir, output_dir, indices):
    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    usr_msg = get_user_query(catalog)

    client = OpenAI()

    for transfer_model in transfer_models:
        transfer_ranks = []

        for index in indices:
            file_path = f"{result_dir}/{base_model}/{dataset}/{catalog}/{index}/random_inference=True.csv"

            if not os.path.exists(file_path):
                print(f"❌ {file_path} does not exist.")
                continue
            
            product_list, target_product, _, _ = get_product_list(catalog, index, dataset)

            df = pd.read_csv(file_path)

            # take the last 5 rows
            df = df.tail(5) # last batch
            ranks = df['product_rank'].tolist()
            min_rank = min(ranks)
            row = df[df['product_rank'] == min_rank]

            attack_prompt = row['attack_prompt'].values[0]
            
            attack_prompt = attack_prompt.replace('<span style="color:red;">', '').replace('</span>', '')

            target_product = product_list[index - 1]['Name']

            random.shuffle(product_list)
            transfer_user_prompt = compose_transfer_prompt(product_list, attack_prompt, target_product, usr_msg)
            
            messages = [
                    {"role": "system", "content": ASSSISTANT_PROMPT},
                    {"role": "user", "content": transfer_user_prompt
                    }
                ]

            completion = client.chat.completions.create(
                    model=transfer_model,
                    messages=messages,
                    max_tokens=500
                )
            
            decoded_output = completion.choices[0].message.content
            # import pdb; pdb.set_trace()
            
            product_names = [product['Name'] for product in product_list]
            rank = rank_products(decoded_output, product_names)[target_product]

            transfer_ranks.append(rank)

        # Save the ranks to a CSV file
        avg_rank = sum(transfer_ranks) / len(transfer_ranks)
        std_rank = statistics.stdev(transfer_ranks)

        results.append({
            'base_model': base_model,
            'transfer_model': transfer_model,
            'dataset': dataset,
            'catalog': catalog,
            "Average Rank": f'{round(avg_rank, 2)}±{round(std_rank, 2)}', 
        })

    df_results = pd.DataFrame(results)
    print(tabulate(df_results, headers='keys', tablefmt='grid'))

    output_dir = f"{output_dir}/{dataset}/{base_model}"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/{catalog}.csv"
    df_results.to_csv(save_path, index=False)
    print(f"✅ Saved to {save_path}")
            

if __name__ == '__main__':
    dataset = 'ragroll'
    catalogs = [
        "air compressor",
        "air purifier",
        "automatic garden watering system",
        "barbecue grill",
        "beard trimmer",
        "blender",
        "coffee maker",
        "computer monitor",
        "computer power supply",
        "cordless drill",
        "curling iron",
        "dishwasher",
        "electric sander",
        "electric toothbrush",
        "eyeshadow",
        "fascia gun",
        "hair dryer",
        "hair straightener",
        "hammock",
        "hedge trimmer",
        "laptop",
        "laser measure",
        "lawn mower",
        "leaf blower",
        "lipstick",
        "microwave oven",
        "network attached storage",
        "noise-canceling headphone",
        "paint sprayer",
        "pool cleaner",
        "portable air conditioner",
        "portable speaker",
        "pressure washer",
        "robot vacuum",
        "screw driver",
        "shampoo",
        "skin cleansing brush",
        "sleeping bag",
        "slow cooker",
        "smartphone",
        "solid state drive",
        "space heater",
        "string trimmer",
        "tablet",
        "tent",
        "tool chest",
        "washing machine",
        "wet-dry vacuum",
        "wifi router",
        "wood router"
    ]

    all_models = [
        'llama-3.1-8b',
        'vicuna-7b',
        'mistral-7b',
        'deepseek-7b'
    ]
    result_dir = f"result/suffix/v1"
    output_dir = f"transfer"
    indices = [1, 2, 3, 4, 5, 6, 7, 8]

    # for base_model in all_models[1]:
    
    import argparse
    parser = argparse.ArgumentParser(description='Transfer Evaluation')
    parser.add_argument('--base', type=str, default='llama-3.1-8b', help='Base model name')
    args = parser.parse_args()

    base_model = args.base

    # transfer_models = [model for model in all_models if model != base_model]
    transfer_models = ['gpt-4.1']
    for catalog in catalogs:
        if os.path.exists(f"{output_dir}/{dataset}/{base_model}/{catalog}.csv"):
            print(f"✅ {output_dir}/{dataset}/{base_model}/{catalog}.csv already exists.")
            continue
        # transfer_evaluate(dataset, catalog, base_model, transfer_models, result_dir, output_dir, indices)
        transfer_gpt(dataset, catalog, base_model, transfer_models, result_dir, output_dir, indices)

    