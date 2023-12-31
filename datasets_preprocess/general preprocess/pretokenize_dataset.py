import argparse
import os
from datetime import datetime
from pathlib import Path

from transformers import GPT2TokenizerFast

from jsonlines import jsonlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="default", help='Dataset to use')
    args = parser.parse_args()

    # Prepare list of datasets
    datasets_folder = "/data/rv.lopes/datasets"
    """
    datasets = ["ptwikidocs-train-clean","arquivodocs-train-clean",
                "europarldocs-train", "oscar-ptpt-train-clean",
                "opensubtitles-filtered"]
    """
    datasets = [args.dataset]
    print(args)

    # Load our tokenizer
    tokenizers_dir = "/data/rv.lopes/tokenizers/"
    loaded_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizers_dir + "gptuga-tk-512")
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    counter = 0
    for d in datasets:
        counter += 1
        # Build dataset dir
        cur_dataset_dir = datasets_folder + "/" + d
        print("Starting token count for dataset: ", d)
        # Get dataset json files
        dataset_file_paths = []
        for file in os.listdir(cur_dataset_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(cur_dataset_dir + "/" + file)
        print("Found files", dataset_file_paths)

        train_save_dir = "/data/rv.lopes/datasets/" + d + "-pretokenized"
        file_prefix = "doc_"
        train_file_dir = train_save_dir + "/" + file_prefix

        if not os.path.exists(train_save_dir):
            Path(train_save_dir).mkdir(exist_ok=True)

        docs_per_file = 100000  # max number of documents per json file (in train split)
        total_train_count = 0  # total number of docs in train split
        file_count = 0  # number of written files
        doc_buffer = []

        print("Starting to tokenize ", d)
        start_time = datetime.now()
        for f in dataset_file_paths:
            with jsonlines.open(f) as reader:
                for doc in reader:
                    doc_tokens = loaded_tokenizer(doc['text'], return_tensors="pt", max_length=512,
                                          truncation=True,
                                          padding="max_length")
                    doc_buffer.append({key: tensor.tolist() for key, tensor in doc_tokens.items()})
                    total_train_count += 1
                    if total_train_count % docs_per_file == 0:
                        with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
                            d.write_all(doc_buffer)
                        doc_buffer = []
                        file_count += 1

        with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
            d.write_all(doc_buffer)

        end_time = datetime.now()
        print('Finished tokenizing to disk. Duration: {}'.format(end_time - start_time))
        #if counter > 0:
            #break
