# snli-artifacts

### Repo Description
This repo contains code for training debiased LLMs for performing Natural Langauge Inference (NLI) tasks. In particular, it uses the techniques of example reweighting, and product of expert losses from "[Towards Debiasing NLU Models from Unknown Biases](https://aclanthology.org/2020.emnlp-main.613/)" to reweight the loss function for each training example based on how biased it is. The key advantage of this technique is that no knowledge of the types of bias contained in the data is needed to be able to use it to debias models. During the training process, the model will learn to avoid using simple strategies that work well on a given dataset but fail to generalize to challenge datasets. 

The code is written in python and uses a mixture of the huggingface and pytorch libraries.

### Getting Started
Clone the repository:

`git clone link`

Install dependencies:

`pip install -r requirements.txt`

Python >= 3.10 is needed use this repo.

### Using this Repo


`./python/run.py` can be used to train and evaluate models (with standard loss) on NLI tasks.

`./python/reweighted_training.py` can be used to train models using example reweighted and product of expert loss functions on NLI tasks.

`./notebook/example.ipynb` provides code examples on how the two above files can be used to train and evaluate a debiased model. 

`./notebook/use_case.ipynb` goes into more background detail about the task of NLI, dataset biases, and why it can be useful to change the loss function by walking through all these topics on the SNLI dataset.  