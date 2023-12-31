import argparse
import os
from datetime import datetime
import jsonlines
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from simhash import Simhash, SimhashIndex
import ftfy
from ftfy import fix_text
import re


# from bs4 import BeautifulSoup


def load_dataset(datasetName, pre_split=False):
    dataset = []
    original_doc_count = 0
    dataset_file_paths = []
    if pre_split:
        # Pre-split means we have an already split dataset in two folders:train-split
        train_dir = "/user/home/rv.lopes/data/datasets/" + datasetName + "-train"
        for file in os.listdir(train_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(train_dir + "/" + file)
        eval_dir = "/user/home/rv.lopes/data/datasets/" + datasetName + "-eval"
        for file in os.listdir(eval_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(eval_dir + "/" + file)
    else:
        dataset_dir = "/user/home/rv.lopes/data/datasets/" + datasetName
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


def remove_pub_prefix(string):
    # Define the regex pattern to match "Pub" at the start of the string
    pattern = r'^Pub'
    # Use the sub() function from the re module to replace the matched pattern with an empty string
    result = re.sub(pattern, '', string)
    return result


def fix_mojibakes(text):
    removed_pub_text = remove_pub_prefix(text)
    return ftfy.fix_text(removed_pub_text)


def remove_html_tags(text):
    # soup = BeautifulSoup(text, 'html.parser')
    # cleaned_text = soup.get_text(separator=' ')
    # return cleaned_text
    return text


def calculate_simhash(text):
    words = text.split()
    # remove stop words before calculating simhash?
    # filtered_words = [word for word in words if word.lower() not in stop_words]
    # filtered_document = ' '.join(filtered_words)
    return Simhash(words)


def remove_duplicates(dataset):
    hashes = [(i, calculate_simhash(doc)) for i, doc in enumerate(dataset)]
    index = SimhashIndex(hashes, k=3)  # Adjust k according to the desired similarity threshold

    filtered_dataset = []
    seen_ids = set()
    c = 0
    for i, doc in enumerate(dataset):
        doc_hash = calculate_simhash(doc)
        near_dups_ids = index.get_near_dups(doc_hash)

        # Make sure the document itself is not detecting it as a near-dup
        for nd in near_dups_ids:
            if int(nd) != i:
                seen_ids.add(int(nd))

        # Only add document if its hash has not been seen before, and its
        if i not in seen_ids:
            seen_ids.add(i)
            filtered_dataset.append(doc)
            index.add(i, doc_hash)
            c += 1
    print("added " + str(c) + " docs when removing duplicates")
    return filtered_dataset


def remove_duplicates_set(dataset):
    seen_texts = set()
    for doc in dataset:
        if doc not in seen_texts:
            seen_texts.add(doc)

    return list(seen_texts)


def remove_documents_by_word_count(dataset, min_words):
    filtered_dataset = []
    for document in dataset:
        words = document.split()
        if len(words) >= min_words:
            filtered_dataset.append(document)
    return filtered_dataset


def remove_stopwords(dataset):
    filtered_dataset = []
    for document in dataset:
        words = document.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_dataset.append(' '.join(filtered_words))
    return filtered_dataset


# from bs4 import BeautifulSoup
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="ptwikicleanv2", help='Dataset to use')
    parser.add_argument('-s', dest='dataset_save', default="ptwikidocs2", help='Dataset to use')
    args = parser.parse_args()

    # Load stopwords
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))

    # Example usage
    datasetName = args.dataset
    min_words = 15

    # Load dataset
    print("Loading dataset " + datasetName)
    start_time = datetime.now()
    dataset = load_dataset(datasetName, pre_split=False)
    # dataset = dataset[:20000]
    end_time = datetime.now()
    print('Load dataset finished. Duration: {}'.format(end_time - start_time))

    do_mojibakes = True
    do_html_clean = False
    do_dup_removal = True
    do_remove_by_word_count = True
    # Fix mojibakes
    if do_mojibakes:
        print("Starting mojibakes cleanup")
        start_time = datetime.now()
        dataset = [fix_mojibakes(doc['text']) for doc in dataset]
        end_time = datetime.now()
        print('Fix mojibakes finished. Duration: {}'.format(end_time - start_time))

    # Remove html tags remnants
    if do_html_clean:
        print("Starting html tags removal")
        start_time = datetime.now()
        dataset = [remove_html_tags(doc['text']) for doc in dataset]
        end_time = datetime.now()
        print('Remove html tags finished. Duration: {}'.format(end_time - start_time))

    if do_remove_by_word_count:
        # Remove docs by word count
        print("Starting docs removal by word count")
        start_time = datetime.now()
        dataset = remove_documents_by_word_count(dataset, min_words)
        end_time = datetime.now()
        print('Remove docs by word count finished. Duration: {}'.format(end_time - start_time))

    if do_dup_removal:
        # Remove dups
        print("Starting duplicate removal")
        start_time = datetime.now()
        dataset = remove_duplicates_set(dataset)  # remove_duplicates(dataset)
        end_time = datetime.now()
        print('Remove dups finished. Duration: {}'.format(end_time - start_time))

    # ----------------------------------------------------- #
    # Dataset split directories
    datasetSave = args.dataset_save
    train_save_dir = "/user/home/rv.lopes/data/datasets/" + datasetSave + "-train-clean"
    eval_save_dir = "/user/home/rv.lopes/data/datasets/" + datasetSave + "-eval-clean"
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
