# controllable-seo
## Setup environment
```conda create -n [name] python=3.11```

```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

```pip install transformers```

```conda install -c conda-forge accelerate```

```conda install anaconda::seaborn```

```conda install conda-forge::openai```

```pip install torch transformers wandb accelerate```

Log in to your huggingface with your authorized token:
```huggingface-cli login```


Make sure to modify wandb's ```ENITY``` and ```PROJECT``` to your project group


## Experiment
Run under the root directory

```python -m experiment.main```

If you want to do a grid search, we used a wandb sweep. Modify the parameters in ```config.yaml``` to values and give a list of hparams you want to search on.

```value``` for a single hparam and ```values``` for a list of hparams
