import random

import math
import os
from itertools import islice

import jsonlines
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer


class SingleDatasetIterable(IterableDataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, name, loaded_tokenizer: PreTrainedTokenizer, seq_len: int,
                 targetDataset: str, docs_per_file: int, debug: bool = False, streaming: bool = False,
                 proc_rank: int = None, num_procs: int = None, pretokenized: bool = False):
        self.name = name
        self.proc_rank = proc_rank
        self.worker_info = None
        self.tokenizer = loaded_tokenizer
        self.maxLength = seq_len
        self.total_docs_seen = 0
        self.seen_data = 0
        self.data = []
        self.streaming = streaming
        self.pretokenized = pretokenized
        # Datasets dir
        self.datasetsFolder = "/data/rv.lopes/datasets"

        # Get target dataset files | targetDataset = ptwikidocs
        self.dir = self.datasetsFolder + "/" + targetDataset
        if self.pretokenized:
            self.dir = self.dir + "-pretokenized"
        self.files = []
        for file in os.listdir(self.dir):
            if file.endswith('.json'):
                self.files.append(self.dir + "/" + file)

        if debug is False:
            if "clueweb-large" in targetDataset:
                self.total_docs = 29009246
            else:
                len_files = len(self.files)
                last_file = self.files[len_files - 1]
                self.total_docs = 0
                if len(self.files) > 1:
                    last_file_num = str(len_files - 1)
                    last_file = ""
                    for f in self.files:
                        if last_file_num in f:
                            last_file = f
                            break

                # Get dataset length without iterating all of it
                num_full_files = len(self.files) - 1  # last one is not guaranteed to have X docs p/ file
                self.total_docs = num_full_files * docs_per_file  # total docs without last file | num_full_files * docs_per_file

                with jsonlines.open(last_file) as reader:  # open last file to count docs
                    for obj in reader:
                        self.total_docs = self.total_docs + 1  # add docs in last file to total count

            if not streaming:
                # Read the entire UNTOKENIZED dataset unto memory
                for f in self.files:
                    with jsonlines.open(f) as reader:
                        for obj in reader:
                            if self.pretokenized:
                                self.data.append(obj)
                            else:
                                self.data.append(obj['text'])
                        # self.data.append([obj['text'] for obj in reader])
        else:
            # If debugging we only read 1 file, and we only get the first 5k from each one
            max_docs = 5000
            read_docs = 0
            self.total_docs = 1 * max_docs  # docs_per_file
            last_file = self.files[0]
            with jsonlines.open(last_file) as reader:
                for obj in reader:
                    if self.pretokenized:
                        self.data.append(obj)
                    else:
                        self.data.append(obj['text'])
                    read_docs += 1
                    if read_docs >= max_docs:
                        break

        # If running in a distributed environment, we want to make sure
        # the dataset is split according to the rank to avoid data
        # duplication along processes.
        if self.proc_rank is not None and num_procs is not None:
            if not streaming:
                per_rank = int(len(self) / num_procs)
                start = proc_rank * per_rank
                end = (proc_rank + 1) * per_rank
                if proc_rank == num_procs - 1:
                    end = len(self)
                self.data = self.data[start:end]
            else:
                per_rank = int(len(self.files) / num_procs)
                start = proc_rank * per_rank
                end = (proc_rank + 1) * per_rank
                if proc_rank == num_procs - 1:
                    end = len(self)
                self.files = self.files[start:end]

    def __len__(self):
        return self.total_docs

    # Returns TRUE if it has iterated all the data
    def has_seen_all(self):
        return self.seen_data == self.total_docs

    def get_seen_data(self):
        return self.seen_data

    def shuffle_data(self, epoch):
        # Reset some vars
        self.seen_data = 0
        if not self.streaming:
            # We shuffle the contents
            # for d in self.data:
            # random.Random(epoch).shuffle(d)
            # Then we shuffle the "files"
            random.Random(epoch).shuffle(self.data)
        else:
            random.Random(epoch).shuffle(self.files)

    def get_iterator(self):
        for entry in self.data:
            # for entry in d:
            self.seen_data += 1
            self.total_docs_seen += 1
            if not self.pretokenized:
                encoding = self.tokenizer(entry, return_tensors="pt", max_length=self.maxLength,
                                          truncation=True,
                                          padding="max_length")
            else:
                encoding = entry
            # Yield each document
            yield {key: tensor for key, tensor in encoding.items()}

    def get_streaming_iterator(self):
        for f in self.files:
            with jsonlines.open(f) as reader:
                for entry in reader:
                    self.seen_data += 1
                    self.total_docs_seen += 1
                    if not self.pretokenized:
                        encoding = self.tokenizer(entry['text'], return_tensors="pt", max_length=self.maxLength,
                                                  truncation=True,
                                                  padding="max_length")
                    else:
                        encoding = entry
                    # Yield each document
                    yield {key: tensor for key, tensor in encoding.items()}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            if self.streaming:
                return self.get_streaming_iterator()
            return self.get_iterator()
        else:
            # Multi process loading
            per_worker = int(len(self) / worker_info.num_workers)
            iter_start = worker_info.id * per_worker
            iter_end = (worker_info.id + 1) * per_worker
            if worker_info.id == worker_info.num_workers - 1:
                iter_end = len(self)
            it = self.get_iterator()
            return islice(it, iter_start, iter_end, 1)
