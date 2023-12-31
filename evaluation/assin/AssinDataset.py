import random

from torch.utils.data import Dataset
from transformers import GPT2ForTokenClassification
import torch


class AssinDataset(Dataset):
    CLASSES = ["None", "Entailment", "Paraphrase"]

    def __init__(self, loaded_tokenizer, batchSize, data, seq_len,
                 categoric=False, multitask=False):
        self.tokenizer = loaded_tokenizer
        self.batchSize = batchSize
        self.maxLength = seq_len
        self.numSamples = len(data)
        self.categoric = categoric
        self.data = data
        self.tokenizedData = []
        for doc in data:
            #eos_token = self.tokenizer.eos_token # +eos_token
            tok_doc = loaded_tokenizer(text=doc['premise'], text_pair=doc['hypothesis'], padding='max_length',
                                     truncation=True, max_length=128,
                                     return_tensors='pt')
            lab = None
            if multitask:
                rte_label = doc['entailment_judgment']
                sts_label = doc['relatedness_score']
                lab = (rte_label, sts_label)
            else:
                if categoric:  # is entailment:
                    lab = doc['entailment_judgment']
                else:
                    lab = doc['relatedness_score']

            self.tokenizedData.append({
                "input_ids": tok_doc['input_ids'].squeeze(),
                "attention_mask": tok_doc['attention_mask'].squeeze(),
                "labels": lab,
            })

    def shuffle(self, epoch):
        random.Random(epoch).shuffle(self.data)

    def __len__(self):
        return self.numSamples #len(self.data)

    def __getitem__(self, idx: int):
        encs = self.tokenizedData[idx]
        return encs
