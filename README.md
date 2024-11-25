# controllable-seo
## Setup environment
```conda create -n [name] python=3.11```

```pip install transformers wandb```

Make sure to modify wandb's ```ENITY``` and ```PROJECT``` to your project group

If you want to do a grid search, we used a wandb sweep. Modify the parameters in ```config.yaml``` to values and give a list of hparams you want to search on.

```value``` for a single hparam and ```values``` for a list of hparams
