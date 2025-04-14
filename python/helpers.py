import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm

QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples

# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }