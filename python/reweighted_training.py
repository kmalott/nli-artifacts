import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, TrainingArguments, HfArgumentParser, \
    AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

NUM_PREPROCESSING_WORKERS = 2

def prepare_dataset_nli_custom(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    tokenized_examples['predicted_scores'] = examples['predicted_scores']
    return tokenized_examples

def main():
    argp = HfArgumentParser(TrainingArguments)
    # Useful Arguments:
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    prepare_train_dataset = lambda exs: prepare_dataset_nli_custom(exs, tokenizer, args.max_length)
    #prepare_train_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    train_dataset = None
    train_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    train_dataset_featurized.set_format('torch')
    train_dataset_featurized = train_dataset_featurized.rename_column("label", "labels")


    #load data with torch dataloader
    batch_size = training_args.per_device_train_batch_size
    dataloader = DataLoader(
        train_dataset_featurized, batch_size = batch_size
    )

    #initialize optimizer, scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = training_args.num_train_epochs
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    #set annealing parameter
    a = 0.8

    #training loop
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    progress_bar = tqdm(range(int(num_training_steps)))
    model.train()
    t = 0
    for epoch in range(int(num_epochs)):
        b = 0
        total_loss = 0
        for batch in dataloader:
            predicted_scores = batch['predicted_scores']
            batch.pop('predicted_scores', None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(logits)
            probs = probs.log()
            predicted_probs = softmax(predicted_scores)
            p_ic = torch.empty((predicted_probs.shape[0]))
            p_d = torch.empty((predicted_probs.shape[0]))
            # for i in range(0, predicted_probs.shape[0]):
            #     p_ic[i] = predicted_probs[i,batch['labels'][i]]
            #     p_d[i] = probs[i,batch['labels'][i]]
            alpha_t = 1 - ((t*(1-a))/num_training_steps)
            for i in range(0, predicted_probs.shape[0]):
                p_ic[i] = torch.pow(predicted_probs[i,batch['labels'][i]], alpha_t) / torch.sum(torch.pow(predicted_probs[i,:], alpha_t))
                p_d[i] = probs[i,batch['labels'][i]]
            loss = -(1-p_ic)*p_d
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            b += 1
            total_loss += loss
            t += 1
        print(f"Average loss on epoch {epoch}: {total_loss / b}")

    #saving model
    model.save_pretrained(training_args.output_dir, from_pt = True)
    tokenizer.save_vocabulary(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()