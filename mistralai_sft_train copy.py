#!/usr/bin/env python
# coding: utf-8

# Fine-Tune Llama2-7b on custom dataset
import os, ipdb
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch, random
from datasets import DatasetDict, Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


# from ../evaluation_metrics import Metrics
seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # or "true", depending on your needs

device = 'cuda' if torch.cuda.is_available() else "cpu"
device

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

# script_args.per_device_train_batch_size,
script_args.gradient_accumulation_steps,
# script_args.per_device_eval_batch_size,

script_args.seq_length

script_args.model_name = "mistralai/Mistral-7B-v0.1"
# script_args.model_name = "meta-llama/Llama-2-7b-hf"
script_args.size = "7b"
# script_args.seq_length = 3000
# # script_args.seq_length = 512
script_args.seq_length = 2400

# script_args.model_name = "meta-llama/Llama-2-13b-hf"
# script_args.size = "13b"
# script_args.seq_length = 3000
# script_args.seq_length = 2400
# script_args.seq_length = 1024

# script_args.dataset_name = "./data/LLLM_DOCTEAT_TDM_ALL_TEMPLATE/fold2"
# script_args.output_dir = "./model_ckpt/docteat_tdm_f2_all_template"
# script_args.run_name = "sft_llama2_docteat_tdm_f2_all_Template"

# script_args.dataset_name = "./data/LLLM_DOCTEAT_TDMS_ALL_TEMPLATE/fold2"
# script_args.output_dir = f"./model_ckpt/docteat_llama2_{script_args.size}_tdms_f2_all_template"
# script_args.run_name = f"sft_docteat_llama2_{script_args.size}_tdms_f2_all_Template"
# script_args.dataset_name = "./data/LLLM_DOCTEAT_TDMS_ALL_TEMPLATE/fold1"
# script_args.output_dir = f"./model_ckpt/docteat_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"
# script_args.run_name = f"sft_docteat_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"

# script_args.per_device_train_batch_size = 2
# script_args.gradient_accumulation_steps = 2
# # multi GPU
# # script_args.per_device_train_batch_size = 4

# i = 1
# script_args.test_dataset = f"../data/LLLM_LONG_SUMMARIZED_TDMS_SQUAD_{i}/fold1"
# script_args.dataset_name = "../data/LLLM_LONG_TDMS_ALL_TEMPLATE/fold1"
script_args.dataset_name = "./data/LLLM_LONG_SUMMARIZED_TDMS_ALL_TEMPLATE/fold1"
script_args.output_dir = f"./model_ckpt/long_summ_mistralai_{script_args.size}_tdms_f1_all_template_seq_{script_args.seq_length}"
script_args.run_name = f"sft_long_summ_mistralai_{script_args.size}_tdms_f1_all_template_seq_{script_args.seq_length}"

script_args.per_device_train_batch_size = 3
script_args.gradient_accumulation_steps = 2
script_args.per_device_eval_batch_size = 2

# ## 50% setup  
# script_args.dataset_name = "data/LLLM_DOCTEAT_TDMS_ALL_TEMPLATE_50_PERCENT/fold1"
# script_args.output_dir = f"./model_ckpt/docteat_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}_50_percent"
# script_args.run_name = f"sft_docteat_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}_50_percent"

# script_args.per_device_train_batch_size = 6
# script_args.gradient_accumulation_steps = 2
# # multi GPU
# script_args.per_device_train_batch_size = 4

# script_args.dataset_name = "./data/LLLM_LONG_TDMS_ALL_TEMPLATE/fold1"
# script_args.output_dir = f"./model_ckpt/long_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"
# script_args.run_name = f"sft_long_llama2_{script_args.size}_tdms_f1_all_template_seq_len_{script_args.seq_length}"

# script_args.per_device_train_batch_size = 2
# script_args.gradient_accumulation_steps = 2
# script_args.per_device_eval_batch_size = 2

script_args.save_steps = 1000
script_args.eval_steps = 1000
script_args.evaluation_strategy = 1000
script_args.logging_steps = 1000
script_args.streaming = False
script_args.num_train_epochs = 5
script_args.save_total_limit = 50

script_args.save_strategy = "steps" 
script_args.evaluation_strategy= "steps" 
# script_args.save_strategy = "epoch"
# script_args.evaluation_strategy= "epoch"


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['answer']}"
    # text = f"{example['prompt']}\n{example['answer']}"
    return text


def create_datasets(tokenizer, args):
    # dataset = load_dataset(
    #     args.dataset_name,
    #     data_dir=args.subset,
    #     split=args.split,
    #     use_auth_token=True,
    #     num_proc=args.num_workers if not args.streaming else None,
    #     streaming=args.streaming,
    # )
    
    dataset = DatasetDict.load_from_disk(f"{args.dataset_name}")
    dataset = dataset.shuffle(seed=seed)
    
    # if args.streaming:
    #     print("Loading the dataset in streaming mode")
    #     valid_data = dataset.take(args.size_valid_set)
    #     train_data = dataset.skip(args.size_valid_set)
    #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    # else:
    
    # dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_data = dataset["train"]
    valid_data = dataset["validation"].shard(num_shards=5, index=0)
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, nb_examples=600)
    # chars_per_token = chars_token_ratio(train_data, tokenizer, nb_examples=len(train_data)//2)
    # 3.70
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token="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",
    # use_auth_token=True,
)

base_model.config.use_cache = False

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, 
    use_auth_token="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",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# tokenizer.add_tokens(AddedToken("{", normalized=False))
# tokenizer.add_tokens(AddedToken("}", normalized=False))


# https://github.com/tatsu-lab/stanford_alpaca/issues/133#issuecomment-1483893538
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    # max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    evaluation_strategy=script_args.evaluation_strategy,
    save_strategy=script_args.save_strategy,
    eval_steps = script_args.eval_steps,
    load_best_model_at_end=True,
    save_total_limit=script_args.save_total_limit,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    # bf16=True,
    fp16=True,
    remove_unused_columns=False,
    num_train_epochs = script_args.num_train_epochs,
    run_name=script_args.run_name,
)


print(torch.__version__)

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

len(train_dataset)

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


expected_steps = ((len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs)// num_gpus
# expected_steps = (len(train_dataset) // (training_args.per_device_train_batch_size)) * training_args.num_train_epochs
print(f"Expected steps: {expected_steps}")

print(f"Max token lenght: {tokenizer.model_max_length}")
print(f"Batch size: {script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps * num_gpus }")
print(f"Number of GPUs available: {num_gpus}")

# print(script_args)|
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    # max_seq_length=None,
    max_seq_length=script_args.seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()


# # MODEL SAVING
# trainer.save_model(script_args.output_dir)
trainer.save_model(f"{script_args.output_dir}/best_checkpoint")

# output_dir = os.path.join(script_args.output_dir, "final_checkpoint_")
# output_dir = os.path.join(script_args.output_dir, f"{script_args.run_name}")
trainer.model.save_pretrained(f"{script_args.output_dir}/save_pretrained")
