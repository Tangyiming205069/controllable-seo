

# ["coffee_machines", "books", "cameras"]
# ['llama-3.1-8b', 'llama-2-7b', 'vicuna-7b', 'mistral-7b', 'deepseek-7b']

# python -m experiment.main --mode=suffix --catalog=coffee_machines --model=llama-3.1-8b
# python -m experiment.main --mode=suffix --catalog=books --model=llama-3.1-8b
# python -m experiment.main --mode=suffix --catalog=cameras --model=llama-3.1-8b

# python -m experiment.main --mode=suffix --catalog=coffee_machines --model=vicuna-7b
# python -m experiment.main --mode=suffix --catalog=books --model=vicuna-7b
# python -m experiment.main --mode=suffix --catalog=cameras --model=vicuna-7b

# python -m experiment.main --mode=suffix --catalog=coffee_machines --model=mistral-7b
# python -m experiment.main --mode=suffix --catalog=books --model=mistral-7b
# python -m experiment.main --mode=suffix --catalog=cameras --model=mistral-7b

# python -m experiment.main --mode=suffix --catalog=coffee_machines --model=deepseek-7b
# python -m experiment.main --mode=suffix --catalog=books --model=deepseek-7b
# python -m experiment.main --mode=suffix --catalog=cameras --model=deepseek-7b

# RAGROLL
python -m experiment.main --mode=suffix --dataset=ragroll --catalog="portable speaker" --model=deepseek-7b

