# controllable-seo
## Setup environment
1. Create your environment with python3.11

```conda create -n seo python=3.11```

2. Activate this environment\

```conda activate seo```

3. Install PyTorch with CUDA

```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

4. Install pip dependencies

```pip install -r requirements.txt```

5. Install seaborn

```conda install anaconda::seaborn```

6. Install openai

```conda install conda-forge::openai```

Log in to your huggingface with your authorized token:

```huggingface-cli login```

Make sure to modify wandb's ```ENITY``` and ```PROJECT``` to your project group


## Experiment
Run under the root directory

```python -m experiment.main```

If you want to do a grid search, we used a wandb sweep. Modify the parameters in ```config.yaml``` to values and give a list of hparams you want to search on.

```value``` for a single hparam and ```values``` for a list of hparams
