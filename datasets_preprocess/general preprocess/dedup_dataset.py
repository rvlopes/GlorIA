import argparse
import os
from datetime import datetime
import jsonlines
import pandas
from datasets import load_dataset
import hashlib


# This is an old version using raw python sets...Due to human-error when handling versioning of files,
# the version that used exact deduplication with text hashing was "lost".
# The following method is an approximation of what was actually used.
def remove_duplicates(dataset):
    unique_texts = {}
    duplicates = []

    for text in dataset:
        # Create a hash for each text entry
        text_hash = hashlib.sha256(text['text'].encode()).hexdigest()

        # Check if the hash is already in the dictionary
        if text_hash not in unique_texts:
            unique_texts[text_hash] = text['text']
        else:
            # We can add the duplicates to another structure to save them later
            # or to just count them
            duplicates.append(text['text'])

    # Filter out duplicates from the dataset
    # unique_dataset = [text['text'] for text in dataset if text not in duplicates]
    unique_dataset = [doc[doc_hash] for doc_hash in unique_texts]

    return unique_dataset

def load_dataset_raw(datasetName, pre_split=False):
    dataset = []
    original_doc_count = 0
    dataset_file_paths = []
    if pre_split:
        # Pre-split means we have an already split dataset in two folders:train-split
        train_dir = "/data/rv.lopes/datasets/" + datasetName + "-train"
        for file in os.listdir(train_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(train_dir + "/" + file)
        eval_dir = "/data/rv.lopes/datasets/" + datasetName + "-eval"
        for file in os.listdir(eval_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(eval_dir + "/" + file)
    else:
        dataset_dir = "/data/rv.lopes/datasets/" + datasetName
        for file in os.listdir(dataset_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(dataset_dir + "/" + file)

    print("Found " + str(len(dataset_file_paths)) + " files for current dataset.")
    for f in dataset_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                dataset.append(obj)
                original_doc_count += 1
    print("Read a total of " + str(original_doc_count) + " documents.")
    return dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="ptwikicleanv2", help='Dataset to use')
    parser.add_argument('-s', dest='dataset_save', default="ptwikidocs2", help='Dataset to use')
    args = parser.parse_args()

    datasetName = args.dataset

    # Load dataset
    print("Loading dataset " + datasetName)
    start_time = datetime.now()
    dataset = load_dataset_raw(datasetName)
    end_time = datetime.now()
    print('Load dataset finished. Duration: {}'.format(end_time - start_time))

    print("Starting duplicate removal")
    start_time = datetime.now()
    # dataset = pandas.DataFrame.from_records(dataset)
    # for d in dataset[:5]:
    #    print(d)
    # dataset = dataset.drop_duplicates(subset="text")
    seen_texts = set()
    for doc in dataset:
        if doc['text'] not in seen_texts:
            seen_texts.add(doc['text'])

    dataset = list(seen_texts)
    seen_texts = set()
    end_time = datetime.now()
    print('Remove dups finished. Duration: {}'.format(end_time - start_time))

    # Dataset split directories
    datasetSave = args.dataset_save
    train_save_dir = "/data/rv.lopes/datasets/" + datasetSave + "-train-clean"
    eval_save_dir = "/data/rv.lopes/datasets/" + datasetSave + "-eval-clean"
    file_prefix = "doc_"
    train_file_dir = train_save_dir + "/" + file_prefix
    eval_file_dir = eval_save_dir + "/" + file_prefix
    # Prep some counters :)
    docs_per_file = 100000  # max number of documents per json file (in train split)
    total_train_count = 0  # total number of docs in train split
    total_eval_count = 0  # total number of docs in eval split
    eval_samples = 10000  # number of examples to put in eval set
    file_count = 0  # number of written files
    total_num_words = 0
    # Prep documents to write
    doc_buffer = []

    print("Starting to write cleaned dataset")
    start_time = datetime.now()
    # Write the dataset into train and eval splits
    for doc in dataset:
        # count words and add them to counter
        num_words = len(doc.split())
        total_num_words += num_words
        # build eval split
        if total_eval_count < eval_samples:
            # prepare doc to add to buffer
            newDoc = {"id": total_eval_count, "text": doc}
            doc_buffer.append(newDoc)
            total_eval_count += 1
            if total_eval_count >= eval_samples:
                with jsonlines.open(eval_file_dir + str(file_count) + ".json", 'a') as d:
                    d.write_all(doc_buffer)
                doc_buffer = []
        # build train split
        else:
            # prepare doc to add to buffer
            newDoc = {"id": total_train_count, "text": doc}
            doc_buffer.append(newDoc)
            total_train_count += 1
            if total_train_count % docs_per_file == 0:
                with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
                    d.write_all(doc_buffer)
                doc_buffer = []
                file_count += 1

    with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
        d.write_all(doc_buffer)

    end_time = datetime.now()
    print('Finished writing cleaned dataset to disk. Duration: {}'.format(end_time - start_time))

    total_docs = total_eval_count + total_train_count
    print("Total train examples: " + str(total_train_count))
    print("Total eval examples: " + str(total_eval_count))
    print("Total number of examples: " + str(total_docs))
    print("Total number of words: " + str(total_num_words))
    print("Average words p/ document: " + str(total_num_words / total_docs))
