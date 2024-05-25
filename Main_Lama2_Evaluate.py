#!/usr/bin/env python
# coding: utf-8

#imports
import pandas as pd
import os, ipdb
import random, evaluate

import random
import string

# Fine-Tune Llama2-7b on custom dataset
import os, ipdb
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch, random
from datasets import DatasetDict, Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer\
, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, TrainerCallback, pipeline

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


# from ../evaluation_metrics import Metrics
seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from evaluation_metrics import Metrics, THRESHOLD

os.environ["TOKENIZERS_PARALLELISM"] = "false" # or "true", depending on your needs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
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

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['answer']}"
    # text = f"{example['prompt']}\n{example['answer']}"
    return text
    

script_args.model_name = "meta-llama/Llama-2-7b-hf"
script_args.size = "7b"
script_args.seq_length = 2400


script_args.save_total_limit = 10
script_args.per_device_train_batch_size = 1
script_args.gradient_accumulation_steps = 1

# script_args.test_ckpt = "checkpoint-76000" # "checkpoint-4000"
script_args.test_ckpt = "checkpoint-5000" # "checkpoint-5000"
i = 1
script_args.test_dataset = f"./data/LLLM_LONG_SUMMARIZED_TDMS_SQUAD_{i}/fold1"
script_args.dataset_name = "./data/LLLM_LONG_SUMMARIZED_TDMS_ALL_TEMPLATE/fold1"
script_args.output_dir = f"./model_ckpt/long_summ_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"
script_args.run_name = f"eval_long_summ_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"

script_args.per_device_train_batch_size = 3
script_args.gradient_accumulation_steps = 2
script_args.per_device_eval_batch_size = 2


script_args.save_steps = 1000
script_args.eval_steps = 1000
script_args.evaluation_strategy = 1000
script_args.logging_steps = 1000
script_args.streaming = False
script_args.num_train_epochs = 5
script_args.save_total_limit = 50

script_args.random_test_sub = 500

script_args.save_strategy = "steps" #"epoch"
script_args.evaluation_strategy= "steps" #"epoch",

model = AutoPeftModelForCausalLM.from_pretrained(
        f"{script_args.output_dir}/{script_args.test_ckpt}",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        use_auth_token="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",
    )

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
        f"{script_args.output_dir}/{script_args.test_ckpt}",
        use_auth_token="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",
    )


# for i in range(1, 8):
#     script_args.test_dataset = f"./data/LLLM_DOCTEAT_TDMS_DROP_{i}/fold1"
for i in range(1, 5):
    script_args.test_dataset = f"./data/LLLM_LONG_SUMMARIZED_TDMS_SQUAD_{i}/fold1"    
    
    dataset = DatasetDict.load_from_disk(f"{script_args.test_dataset}")
        
    valid_data = dataset["validation"].shuffle(seed=42)

    labels = []
    preds = []
    idx_skip = []

    for idx, valid_ex in tqdm(enumerate(valid_data), total=len(valid_data)):
    
        prompt = f"Question: {valid_ex['prompt']}"
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        if inputs.shape[1] >= 5500:
            print(f"Validation index {idx} skipped because input.shape: {inputs.shape}, input split length: {len(valid_ex['prompt'].split())}")
            idx_skip.append(idx)
            continue 

        print(f"inputs.shape: {inputs.shape}, input split length: {len(valid_ex['prompt'].split())}")

        generate_kwargs = dict(
            input_ids=inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )
        
        outputs = model.generate(**generate_kwargs)
        predictions = tokenizer.decode(outputs[0])
        
        preds.append(predictions.split("Answer: ")[-1].replace("</s>", ""))
        labels.append(valid_ex['answer'])
        
        # if idx == len(valid_data)//2 :
        #     results = Metrics.evaluate_property_wise_json_based(label_list=labels, prediction_list=preds)
        #     results.update(Metrics.evaluate_rouge(label_list=labels, prediction_list=preds))
        #     print(f"Intermediate Results:")
        #     for key, value in results.items():
        #         print(f"{key}: {value}")
    
    results = Metrics.evaluate_property_wise_json_based(label_list=labels, prediction_list=preds)
    results.update(Metrics.evaluate_rouge(label_list=labels, prediction_list=preds))
    
    print(f"Test data {script_args.test_dataset}")
    print(f"Test ckpt {script_args.test_ckpt}")
    print(f"Partial THRESHOLD {THRESHOLD}")
    print(f"Total index skipped {len(idx_skip)}")
    print(f"Index skipped {idx_skip}")
    
    print("##################################################################################")
    print(f"Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("##################################################################################")