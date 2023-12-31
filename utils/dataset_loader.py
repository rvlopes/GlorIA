import os
from datasets import load_dataset
from pathlib import Path
import pprint as pp


# RESPONSIBLE FOR LOADING A GIVEN DATASET

# Supports multiple json files for a single dataset
class DatasetLoader:

    def __init__(self, datasetName, benchmark=False):
        self.datasetName = datasetName
        if benchmark:
            self.datasetsFolder = "/data/rv.lopes/benchmarks"
        else:
            self.datasetsFolder = "/data/rv.lopes/datasets"  # "./datasets"
        self.dataset = None
        self.splitSeed = 256879321

    def loadDataset(self):
        datasetDir = self.datasetsFolder + "/" + self.datasetName
        file_paths = []
        for file in os.listdir(datasetDir):
            if file.endswith('.json'):
                file_paths.append(file)
        pp.pprint(file_paths)
        self.dataset = load_dataset(path=datasetDir, split="train", data_files=file_paths)
        print(f"##### Dataset {self.datasetName} loaded. #####")
        return self.dataset

    # Use this to debug/check values: pp.pprint(splitDataset['train']['text'][0])
    # Split dataset into test/eval and training
    # Calculate percentage for eval split - regra 3 simples :)
    def getSplits(self, printSplits):
        targetEvalSize = 4000
        targetEvalPercentage = targetEvalSize / self.dataset.num_rows
        print("Eval percentage is " + str(targetEvalPercentage))
        splitDataset = self.dataset.train_test_split(test_size=targetEvalPercentage, seed=self.splitSeed)

        if printSplits:
            pp.pprint("### DATASET SPLITS ###")
            print("Training Split")
            pp.pprint(splitDataset["train"])  # 95
            print("Evaluation Split")
            pp.pprint(splitDataset["test"])  # 5

        return splitDataset


class DatasetLoaderAccelerate:

    def __init__(self, datasetName, benchmark=False):
        self.datasetName = datasetName
        if benchmark:
            self.datasetsFolder = "/data/rv.lopes/benchmarks"
        else:
            self.datasetsFolder = "/data/rv.lopes/datasets"  # "./datasets"
        self.splitSeed = 256879321

    def loadDataset(self, streaming=False, benchmarkSplit=None, field=None):
        if benchmarkSplit is None:
            datasetDir = self.datasetsFolder + "/" + self.datasetName
        else:
            datasetDir = self.datasetsFolder + "/" + self.datasetName + "/" + benchmarkSplit
        file_paths = []
        for file in os.listdir(datasetDir):
            if file.endswith('.json'):
                file_paths.append(file)
        pp.pprint(file_paths)
        if field is not None:
            dataset = load_dataset(path=datasetDir, field=field, streaming=streaming, split="train", data_files=file_paths)
        else:
            dataset = load_dataset(path=datasetDir, streaming=streaming, split="train", data_files=file_paths)
        print(f"##### Dataset {self.datasetName} loaded. #####")
        return dataset

