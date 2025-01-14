# snli-artifacts

### Repo Description
This repo contains code for training debiased NLI models. In particular, it uses the technique of example reweighting from "[Towards Debiasing NLU Models from Unknown Biases](https://aclanthology.org/2020.emnlp-main.613/)" to reweight the loss function for each training example based on how biased it is. The key advantage of this technique is that no knowledge of the types of bias contained in the data is needed to be able to use it to debias models. During the training process, the model will learn to avoid using simple strategies that work well a given dataset but fail to generalize to challenge datasets. 

The code is written in python and uses a mixture of the huggingface and pytorch libraries.

### Getting Started
Clone the repository:

`git clone link`

Install dependencies:

`pip install -r requirements.txt`

Python >= 3.10 is needed use this repo.

### Using this Repo


`./python/run.py` can be used to train (standard loss) and evaluate models on NLI tasks.

`./python/reweighted_training.py` can be used to train models using example reweighted loss function on NLI tasks.

`./notebook/example.ipynb` demonstrates how the two above files can be used to train and evaluate a model with reweighted loss function. 

`./notebook/use_case.ipynb` goes into more background detail about NLI, dataset biases, and why it can be useful to change the loss function by walking through an example use-case. 