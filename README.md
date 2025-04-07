# controllable-seo
## Setup environment
1. Create your environment with python3.11

```conda create -n seo python=3.11```

2. Activate this environment

```conda activate seo```

3. Install pip dependencies

```pip install -r requirements.txt```

Make sure to modify wandb's ```ENITY``` and ```PROJECT``` to your project group


## Experiment
Run under the root directory

```python -m experiment.main```

If you want to do a grid search, we used a wandb sweep. Modify the parameters in ```config.yaml``` to values and give a list of hparams you want to search on.

```value``` for a single hparam and ```values``` for a list of hparams

## Evaluation
Run main evaluation by 

```python -m experiment.evaluate --job=rank_perplexity```

or 

```python -m experiment.evaluate --job=bad_word```

Bad word ratio on ablation is quite different. Run ```ablatioin_bad_word.py``` file.


