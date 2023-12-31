from torch.utils.data import Dataset
import torch


class SquadV1Dataset(Dataset):

    def __init__(self, loaded_tokenizer, dataset, seq_len):
        self.tokenizer = loaded_tokenizer
        self.maxLength = seq_len
        self.tokenizedData = []
        for entry in dataset:
            eos_token = self.tokenizer.eos_token
            context = entry['context']
            question = entry['question']
            answers = entry['answers']['text']  # Could have the same answer written in diff ways [a1, a2,..]
            tok_context_question = loaded_tokenizer(text="SQUAD Contexto: " + context + ". Pergunta: " + question,
                                       padding='max_length',
                                       truncation=True, max_length=seq_len,
                                       return_tensors='pt')

            tok_answer = loaded_tokenizer(text=answers[0],
                                       padding='max_length',
                                       truncation=True, max_length=seq_len,
                                       return_tensors='pt')

            self.tokenizedData.append({
                "input_ids": tok_context_question['input_ids'].squeeze(),
                "attention_mask": tok_context_question['attention_mask'].squeeze(),
                "labels": tok_answer['input_ids'].squeeze()
            })

    def __len__(self):
        return len(self.tokenizedData)

    def __getitem__(self, idx: int):
        encs = self.tokenizedData[idx]
        return encs
