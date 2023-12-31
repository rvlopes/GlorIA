import os
from typing import List

import jsonlines
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import PreTrainedTokenizer

from accelerate_pipe.single_dataset import SingleDatasetIterable

# https://wandb.ai/wandb_fc/pytorch-image-models/reports/An-Introduction-to-HuggingFace-s-Accelerate-Library--Vmlldzo2MzgzNzA
pre_seeds = [200, 201, 202, 203, 204, 205, 206, 208, 209, 210]


class MultiDataset(Dataset):
    def __init__(self, datasets: List[SingleDatasetIterable],  # [ptwiki dataset, arquivo dataset, clueweb dataset]
                 weights, infiniteMode=False):
        self.infinite_mode = infiniteMode
        self.worker_info = None
        self.datasets = datasets
        self.original_weights = weights.copy()
        self.weights = weights.copy()  # these are our insider weights, which may be updated when a dataset runs out of data
        self.total_size = 0
        self.datasets_exhausted = False
        self.datasets_its = []
        self.has_data = []
        self.datasets_len = []
        self.shuffle_counters = []
        # Count total dataset size and initialize each dataset iterator
        for idx, d in enumerate(self.datasets):
            d_len = len(d)
            self.datasets_len.append(d_len)
            self.total_size = self.total_size + d_len
            self.has_data.append(True)
            self.datasets_its.append(iter(d))
            self.shuffle_counters.append(0)

    def initialize_datasets(self, worker_info):
        self.datasets_its = []
        for d in self.datasets:
            d.worker_info = worker_info
            self.datasets_its.append(iter(d))

    # Pre-processed value hardcoded here?
    # for train split, should be the total lenght of
    # d1['train']+d2['train], for example.
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx: int):
        if self.infinite_mode:
            return self.getitem_infinite(idx)
        else:
            return self.getitem_exhausted_mode(idx)

    def getitem_infinite(self, idx: int):
        target_it = self.datasets_its[idx]
        try:
            next_item = next(target_it)
            return next_item
        except StopIteration:
            # Refresh/recreate exhausted dataset
            shuffle_counter = self.shuffle_counters[idx]
            if shuffle_counter >= len(pre_seeds):
                shuffle_counter = 0
            self.shuffle_counters[idx] = self.shuffle_counters[idx] + 1
            self.datasets[idx].shuffle_data(pre_seeds[shuffle_counter])
            new_iter = iter(self.datasets[idx])
            self.datasets_its[idx] = new_iter
            return next(new_iter)

    # Using this in a dataloader with a weighted sampler,
    # the IDXs we receive here should be the IDs of the
    # dataset. For a batch of 4, dataloader generates [1,1,0,1]
    # where 0=ptwiki and 1=arquivo. This means we have to get
    # our samples sequentially. For the first example, we use idx=1
    # to go to the arquivo dataset and retrieve the next sample available.
    # This keeps going until we exaust it.
    # This would only work if we set the current len in this dataset.
    # If we were to use specific ptwiki and arquivo dataset class files,
    # which we will surely need, we need to decide between normal dataset
    # or iterable. normal dataset requires us to sample a new in-dataset
    # id to return a certain sample. Iterable allows us to get a sample
    # sequentially one by one.
    def getitem_exhausted_mode(self, idx: int):
        try:
            if self.datasets_exhausted:
                raise StopIteration
            elif self.has_data[idx]:
                try:
                    # print(idx)
                    target_it = self.datasets_its[idx]
                    next_item = next(target_it)
                    return next_item
                except StopIteration:
                    next_item = self.dataset_exhausted(idx)
                    return next_item
            else:
                # We do our own sampling. Weights should be updated once we arrive here
                sampler = self.get_sampler()
                # Should only output 1 idx
                for i in sampler:
                    new_idx = i
                # print("Our sampled new id: " + str(new_idx))
                target_it = self.datasets_its[new_idx]
                try:
                    next_item = next(target_it)
                except StopIteration:
                    next_item = self.dataset_exhausted(new_idx)
                return next_item
        except StopIteration:
            # Should never happen
            print("BOMBAAAAAA")
            print(str(idx))
            print(self.has_data)
            for d in self.datasets:
                print(str(d.total_docs))
                print(str(d.seen_data))
                print("################")
            raise StopIteration

    def dataset_exhausted(self, idx):
        print("DATASET " + str(idx) + " HAS RAN OUT OF DATA.")
        # We need to "discard" used dataset
        self.has_data[idx] = False

        # Check if all datasets have become exhausted
        c = 0
        for hd in self.has_data:
            if hd is False:
                c += 1
        if c == len(self.datasets):
            self.datasets_exhausted = True

        # print("Datasets exhausted: " + str(self.datasets_exhausted))
        # print("Has Data List:")
        # print(self.has_data)
        # And we update our inner weights
        self.weights = self.update_weights(idx)
        # print("NEW WEIGHTS!")
        # print(self.weights)
        # Now we should have weights like this [0.4, 0.0, 0.6]
        # So our inner sampler will only output 0 or 2
        # Now we do our own sample
        sampler = self.get_sampler()
        # Should only output 1 idx
        for i in sampler:
            new_idx = i
        # print("Sampling from dataset " + str(new_idx))
        target_it = self.datasets_its[new_idx]
        return next(target_it)

    # Backup sampler with updated weights for when a dataset
    # runs out of samples
    def get_sampler(self):
        weights = self.weights
        # we only need it to output 1, since the main sampler
        # inside the dataloader is the one controlling the number
        # of times getitem is called. this sampler is only used
        # internally for when the other sampler's weights become
        # deprecated, but we still want its internal samples count
        sampler = WeightedRandomSampler(weights, 1)
        return sampler

    def update_weights(self, idx):
        # Now we want to update the weights
        # We prepare new weights
        new_weights = []

        # We set the weights with the total length
        # of the remaining datasets
        for i, d in enumerate(self.datasets):
            if idx == i or self.has_data[i] is False:
                d_len = 0
            else:
                d_len = self.datasets_len[i]
            new_weights.insert(idx, d_len)

        # Here, new_weights should be something like [len(d1), 0, len(d3)], in a case where d2
        # has run out of samples. And so we normalize them.
        new_weights = torch.tensor(new_weights, dtype=float)
        new_weights /= new_weights.sum()
        return new_weights.tolist()

    def reset_dataset(self):
        self.datasets_exhausted = 0
        self.weights = self.original_weights.copy()
        self.datasets_its = []
        self.has_data = []
        for d in self.datasets:
            self.has_data.append(True)
            self.datasets_its.append(iter(d))

    def print_stats(self):
        for idx, d in enumerate(self.datasets):
            print("DATASET " + str(idx))
            print("Total Docs: " + str(len(d)))
            print("Total Docs seen during training: " + str(d.total_docs_seen))
            print("##############################")

    def save_stats(self, checkpoint, rank, path):
        data_stats = []
        for idx, d in enumerate(self.datasets):
            data_stats.append(
                {"checkpoint": checkpoint, "dataset": idx, "total_docs": len(d), "seen_docs": d.total_docs_seen})
        with jsonlines.open(path + "/rank_" + str(rank) + "-datastats" + ".json", 'a') as d:
            d.write_all(data_stats)
