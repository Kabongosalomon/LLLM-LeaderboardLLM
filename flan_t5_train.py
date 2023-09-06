#!/usr/bin/env python
# coding: utf-8

import os, ipdb
import random

import numpy as np
import torch

import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

from datasets import DatasetDict, Dataset, load_from_disk
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
from transformers.optimization import Adafactor, AdafactorSchedule

import random, evaluate

# from ../evaluation_metrics import Metrics


seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # or "true", depending on your needs

# pd.options.display.max_rows , pd.options.display.max_columns  = 100,100  

device = 'cuda' if torch.cuda.is_available() else "cpu"
device

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="google/flan-t5", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses([])[0]


if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")
# script_args.per_device_train_batch_size,
script_args.gradient_accumulation_steps,
# script_args.per_device_eval_batch_size,

script_args.seq_length

# script_args.dataset_name = "./data/LLLM_TDMS_ALL_TEMPLATE/fold1"
# script_args.output_dir = "./model_ckpt/tdms_all_template_v2"
# script_args.run_name = "sft_llama2_tdms_all_Template_v2"

# script_args.dataset_name = "./data/LLLM_DOCTEAT_TDM_ALL_TEMPLATE/fold2"
# script_args.output_dir = "./model_ckpt/docteat_tdm_f2_all_template"
# script_args.run_name = "sft_llama2_docteat_tdm_f2_all_Template"

script_args.model_name = "google/flan-t5"
script_args.size = "large"

script_args.dataset_name = "./data/LLLM_DOCTEAT_TDMS_ALL_TEMPLATE/fold2"
script_args.output_dir = f"./model_ckpt/docteat_flan_t5_{script_args.size}_tdms_f2_all_template"
script_args.run_name = f"sft_docteat_flan_t5_{script_args.size}_tdms_f2_all_Template"

# script_args.dataset_name = "./data/LLLM_DOCTEAT_TDM_ALL_TEMPLATE/fold2"
# script_args.output_dir = f"./model_ckpt/docteat_flan_t5_{script_args.size}_tdm_f2_all_template"
# script_args.run_name = f"sft_docteat_flan_t5_{script_args.size}_tdm_f2_all_Template"

script_args.seq_length = 512
script_args.per_device_train_batch_size = 2
script_args.gradient_accumulation_steps = 2
script_args.per_device_eval_batch_size = 2
script_args.max_source_length = 512
script_args.max_target_length = 512
script_args.label_pad_token_id = -100
script_args.pad_to_multiple_of = 8
script_args.model_max_length = 512

# # multi GPU
# script_args.per_device_train_batch_size = 4

# script_args.dataset_name = "./data/LLLM_LONG_TDM_ALL_TEMPLATE/fold1"
# script_args.output_dir = "./model_ckpt/long_tdm_f1_all_template"
# script_args.run_name = "sft_llama2_long_tdm_f1_all_Template"
# script_args.seq_length = 2400
# script_args.per_device_train_batch_size = 2
# script_args.gradient_accumulation_steps = 2

script_args.save_steps = 50
script_args.logging_steps = 50
script_args.streaming = False
script_args.num_train_epochs = 5
script_args.save_total_limit = 10


tokenizer = AutoTokenizer.from_pretrained(f"{script_args.model_name}-{script_args.size}")

tokenizer.add_tokens(AddedToken("\n", normalized=False))
tokenizer.add_tokens(AddedToken("{", normalized=False))
tokenizer.add_tokens(AddedToken("}", normalized=False))

model = AutoModelForSeq2SeqLM.from_pretrained(f"{script_args.model_name}-{script_args.size}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=script_args.label_pad_token_id,
    pad_to_multiple_of=script_args.label_pad_token_id
)

print(f"Max token lenght: {tokenizer.model_max_length}")

num_gpus = torch.cuda.device_count()

print(f"Max token lenght: {tokenizer.model_max_length}")
print(f"Batch size: {script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps * num_gpus }")
print(f"Number of GPUs available: {num_gpus}")

dataset = DatasetDict.load_from_disk(f"{script_args.dataset_name}")

dataset = dataset.shuffle(seed=seed)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

def compute_metrics(eval_preds):

    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)  # type: ignore
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # type: ignore
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    results = clf_metrics.compute(predictions=[1 if "unanswerable" == x.replace("</s>", "") else 0 for x in decoded_preds], 
                        references=[1 if "unanswerable" == x else 0 for x in decoded_labels]
    )

    rouge = evaluate.load('rouge')
    
    rouge_results = rouge.compute(
        predictions=[pred.replace("</s>", "") for pred in decoded_preds],
        references=decoded_labels
    )
    results.update(rouge_results) 

    # ipdb.set_trace()
    
    # result = Metrics.evaluate_property_wise_text_based(label_list=decoded_labels, prediction_list=decoded_preds)
    # result.update(Metrics.evaluate_rouge(label_list=decoded_labels, prediction_list=decoded_preds))
    return results

def tokenize_function(sample):
    # tokenize inputs
    model_inputs = tokenizer(sample["prompt"], max_length=script_args.max_source_length, 
                             padding="max_length", truncation=True,
                             return_tensors="pt")

    
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["answer"], max_length=script_args.max_target_length, padding="max_length",
                       truncation=True, return_tensors="pt")

    # # Check if the length of labels is >= 512
    # if any(len(label) >= 512 for label in labels["input_ids"]):
    #     return {}  # Return empty dict to skip this example

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]  # type: ignore
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True,
                                            # remove_columns=dataset_columns_to_remove
                                            )
eval_tokenized_dataset = eval_dataset.map(tokenize_function, batched=True,
                                        #   remove_columns=dataset_columns_to_remove
                                          )
print(f"Keys of tokenized dataset: {list(train_tokenized_dataset.features)}")

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)


training_args = Seq2SeqTrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    # optim=script_args.optimizer_type
    num_train_epochs=script_args.num_train_epochs,
    run_name=script_args.run_name,
    
    predict_with_generate=True,
    generation_max_length=script_args.max_target_length,
    
    load_best_model_at_end=True,
    # metric_for_best_model=metric_name,
    # greater_is_better=True,
    
    # logging_dir=f"{model_save_path}/logs",
    # eval_steps=500,  # Evaluate the model every 500 steps,
    evaluation_strategy="epoch",
    # logging_strategy="steps",
    save_strategy="epoch", # steps
    # push_to_hub=False,    
    # seed=seed
)


# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    # max_seq_length=script_args.seq_length,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=([early_stopping_callback])

)

trainer.train()

# # MODEL SAVING
trainer.save_model(script_args.output_dir)

# output_dir = os.path.join(script_args.output_dir, "final_checkpoint_")
output_dir = os.path.join(script_args.output_dir, f"{script_args.run_name}")

trainer.model.save_pretrained(output_dir)

tokenizer.save_pretrained(f"{output_dir}_tokenizer")
best_ckpt_path = trainer.state.best_model_checkpoint
print(f"best epoch: {best_ckpt_path}")

