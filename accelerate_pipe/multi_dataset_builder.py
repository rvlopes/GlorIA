import os

from datasets import load_dataset
from jsonlines import jsonlines
from torch.utils.data import Dataset


class MultiDatasetBuilder:
    def __init__(self, datasets):
        self.datasetsFolder = "/user/home/rv.lopes/data/datasets"
        self.dataFiles = []

        # Get file paths for all the files for each dataset
        for d in datasets:
            directory = self.datasetsFolder + "/" + d
            for json_file in os.listdir(directory):
                if json_file.endswith('.json'):
                    self.dataFiles.append(directory + "/" + json_file)

    def load_and_shuffle_dataset(self):
        # Read dataset and shuffle
        dataset = load_dataset('json', data_files=self.dataFiles)
        dataset = dataset['train'].shuffle()
        dataset.remove_columns(['id'])
        print(dataset[0])
        return dataset.with_format("torch")


class MultiDatasetWithoutWeights(Dataset):
    def __init__(self, datasets, tokenizer, maxLength):
        self.datasetsFolder = "/user/home/rv.lopes/data/datasets"
        self.dataFiles = []
        self.docs = []
        self.tokenizer = tokenizer
        self.maxLength = maxLength

        # Get file paths for all the files for each dataset
        for d in datasets:
            directory = self.datasetsFolder + "/" + d
            for json_file in os.listdir(directory):
                if json_file.endswith('.json'):
                    self.dataFiles.append(directory + "/" + json_file)

        # Read entire untokenized dataset unto memory
        for f in self.dataFiles:
            with jsonlines.open(f) as reader:
                self.docs.append([obj['text'] for obj in reader])

    def __getitem__(self, idx):
        currDoc = self.docs[idx]
        inputs = self.tokenizer(currDoc, return_tensors="pt", max_length=self.maxLength, truncation=True,
                                padding="max_length")
        inputs['labels'] = inputs.input_ids.detach().clone()
        out = {key: tensor[0] for key, tensor in inputs.items()}
        return out

    def __len__(self):
        return len(self.docs)
