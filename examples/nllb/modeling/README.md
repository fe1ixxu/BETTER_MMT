# NLLB Modeling

This subdirectory contains code associated with preprocessing training data, training MT models, generation and computing evaluation metrics for MT models
Installation:

Since this branch uses Hydra 1.1, we'll need to create a new Python environment
```
module load anaconda3/5.0.1
module load cudnn/v8.0.3.33-cuda.11.0
module load cuda/11.0
module load openmpi/4.1.0/cuda.11.0-gcc.9.3.0

# one-time setup:
FAIRSEQ_ENV=/private/home/myleott/.conda/envs/fairseq-20210318-py38
conda create --clone $FAIRSEQ_ENV -n fairseq-20210318_nllb

# to activate the environment:
source activate fairseq-20210318_nllb
```
Make sure you run all your scripts inside the fairseq-20210318_nllb environment.
Install fairseq and hydra 1.1 in the envinroment:
```
cd ~/fairseq-py
pip install --editable .
pip install hydra-core --upgrade --pre
```

Now, try running the demo script:

data preparation
```
python examples/nllb/modeling/prepare_data.py
```

training pipeline
```
python examples/nllb/modeling/train.py \
    --dry-run
```

***
Syntax of training config YAML file:

single value

```
max_update: 10
weight_decay: 0.001
adam_betas: "'(0.1)'"       # enclose by a single quote plus a double quote
langs: "'cs,de,en'"
lr: [0.01]
```

sweeping values
(comma separated values in strings)

```
max_update: 10,20
weight_decay: 0.001
adam_betas: "'(0.9, 0.98)','(0.7, 0.98)'"
optimizer: adam,adadelta
lr: '[0.01],[0.1]'          # enclose by single quote
```
