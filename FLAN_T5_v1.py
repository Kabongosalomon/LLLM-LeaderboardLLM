#!/usr/bin/env python
# coding: utf-8

#imports
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import numpy as np

from collections import defaultdict
import ipdb, os, random, copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from tokenizers import AddedToken

from torch.optim import AdamW
import argparse
from transformers import (
    get_linear_schedule_with_warmup
  )

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer, 
    LongT5Model
) 

device  = 'cuda' if torch.cuda.is_available() else "cpu"

mode = "tdm"
# mode = "tdms"
train_path = f'data/train_{mode}_f1_v2_short.parquet' 
validation_path = f'data/dev_{mode}_f1_v2_short.parquet'

# Check the data 
model_name = ["google/flan-t5", "google/long-t5"]
size = ["-base", "-large", "-xl"]
model_attention = ["","-local", "-tglobal"]

model_idx = 0
size_idx = 0
model_idx = 0

bs = 4
epochs = 5
gpus = -1
workers = os.cpu_count()

os.environ["TOKENIZERS_PARALLELISM"] = "false" # or "true", depending on your needs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'

# model_idx = 1
# size_idx = 0
# model_idx = 1

# # model_max_length = None
# model_max_length = 6000
# max_len_inp = 3500
# model_max_len_out = 2500

model_max_length = 512
# max_len_inp = 432
# max_len_inp = 500
max_len_inp = 512
model_max_len_out = 512

t5_tokenizer = AutoTokenizer.from_pretrained(f"{model_name[model_idx]}{model_attention[model_idx]}{size[size_idx]}", model_max_length=model_max_length)
t5_model = T5ForConditionalGeneration.from_pretrained(f"{model_name[model_idx]}{model_attention[model_idx]}{size[size_idx]}")

t5_tokenizer.add_tokens(AddedToken("{", normalized=False))
t5_tokenizer.add_tokens(AddedToken("}", normalized=False))

print(f"Max token lenght: {t5_tokenizer.model_max_length}")

class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len_inp=512,max_len_out=96):
        self.path = filepath

        self.passage_column = "context"
        self.answer = "answer"
        self.question = "question"
        self.template_question = "template_question"

        # self.data = pd.read_csv(self.path)
        self.data = pd.read_parquet(self.path)

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  #squeeze to get rid of the batch dimension
        target_mask = self.targets[index]["attention_mask"].squeeze()  # convert [batch,dim] to [dim] 

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for rownum, val in tqdm(self.data.iterrows(), total=len(self.data)): # Iterating over the dataframe
            template_question, answer = val[self.template_question], val[self.answer]
         
            input_ = f"{str(template_question)}" # T5 Input format for question answering tasks 
            target = f"{str(answer)}" # Output format we require

            # TODO: Not sure if this is needed as the tokenizer can truncate the output. 
            encoded = t5_tokenizer.batch_encode_plus(
                [input_], 
                truncation = False,
                return_tensors="pt"
            )
            
            if len(encoded['input_ids'][0]) > self.max_len_output:
                continue  
                
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], 
                max_length=self.max_len_input,
                padding='max_length',
                truncation = True,
                return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], 
                max_length=self.max_len_output,
                padding='max_length',
                truncation = True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# In[6]:

train_dataset = QuestionGenerationDataset(t5_tokenizer, train_path, 
                                          max_len_inp=max_len_inp, max_len_out=model_max_len_out)
validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_path, 
                                               max_len_inp=max_len_inp, max_len_out=model_max_len_out)

# In[7]:

# In[8]:


import pytorch_lightning as pl
from torch.optim import AdamW
import argparse
from transformers import (
    get_linear_schedule_with_warmup
  )

class T5Tuner(pl.LightningModule):

    def __init__(self,t5model, t5tokenizer,batchsize=4):
        super().__init__()
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.batch_size = batchsize

    def forward( self, input_ids, attention_mask=None, 
                decoder_attention_mask=None, 
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
         
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss",loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(validation_dataset, 
                          batch_size=self.batch_size,
                          num_workers=2)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


# In[9]:

# In[ ]:

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='min'
)


model = T5Tuner(t5_model, t5_tokenizer, bs)

trainer = pl.Trainer(max_epochs = epochs,
                    #  gpus=gpus,
                     # gpus=-1,
                     strategy = 'auto',
                     accelerator = "auto", 
                     devices = "auto",
                    #  accelerator='gpu', 
                    #  devices=2,
                     callbacks = [
                         early_stop_callback
                     ])

trainer.fit(model)

model.model.save_pretrained(f'model_ckpt/{mode}_{model_name[model_idx].replace("/","_").replace("-","_")}{model_attention[model_idx].replace("-","_")}{size[size_idx].replace("-","_")}_{bs}_{epochs}_{model_max_length}_{max_len_inp}_{model_max_len_out}')
# t5_tokenizer.save_pretrained(f'{model[model_idx]}{model_attention[model_idx]}{size[size_idx]}_tokenizer_{bs}_{epochs}_{model_max_length}_{max_len_inp}_{model_max_len_out}')
t5_tokenizer.save_pretrained(f'model_ckpt/{mode}_{model_name[model_idx].replace("/","_").replace("-","_")}{model_attention[model_idx].replace("-","_")}{size[size_idx].replace("-","_")}_tokenizer_{bs}_{epochs}_{model_max_length}_{max_len_inp}_{model_max_len_out}')
