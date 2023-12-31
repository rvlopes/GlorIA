import random
from datetime import datetime
import gzip
import multiprocessing
import json
import os
import jsonlines
import ftfy
import pprint as pp
import uuid


def generate_short_id():
    uid = uuid.uuid4()
    short_id = uid.hex[:6]  # Take the first 8 characters
    return short_id


def get_clueweb_dir():
    train_save_dir = "/user/home/rv.lopes/data/datasets/clueweb-large-ptpt-train-2/"
    eval_save_dir = "/user/home/rv.lopes/data/datasets/clueweb-large-ptpt-eval/"
    return train_save_dir, eval_save_dir


def read_json_file(file_paths):
    train_dir, eval_dir = get_clueweb_dir()
    # Init variables
    docs = []
    written_file_count = 0
    num_docs = 0
    total_words = 0
    docs_per_file = 100000

    # Get the process ID
    process_id = multiprocessing.current_process().pid

    # Read the JSON file and return its contents
    for f in file_paths:
        with gzip.open(f, 'r') as reader:
            j_reader = jsonlines.Reader(reader)
            for doc in j_reader:
                if ".pt" in doc['URL']:
                    doc_words = len(doc['Clean-Text'].split())
                    total_words += doc_words
                    if doc_words >= 15:
                        cleaned_text = ftfy.fix_text(doc['Clean-Text'].strip())
                        doc_id = str(process_id) + "-" + str(num_docs)
                        newDoc = {"id": doc_id, "text": cleaned_text}
                        docs.append(newDoc)
                        num_docs += 1
                        if num_docs % docs_per_file == 0:
                            written_doc_sufix = generate_short_id()
                            # written_doc_sufix = str(process_id) + "-" + str(written_file_count)
                            print("Writting file ", written_doc_sufix)
                            with jsonlines.open(train_dir + "doc_" + written_doc_sufix + ".json", 'a') as d:
                                d.write_all(docs)
                            docs = []
                            written_file_count += 1
                            print("Writing file ", written_doc_sufix)

    if written_file_count > 0:
        num_docs = num_docs + len(docs)  # add count of missing documents
    else:
        num_docs = len(docs)
    written_doc_sufix = generate_short_id()
    # written_doc_sufix = str(process_id) + "-" + str(written_file_count)
    print("Writting last file in process ", written_doc_sufix)
    with jsonlines.open(train_dir + "doc_" + written_doc_sufix + ".json", 'a') as d:
        d.write_all(docs)
    docs = []
    written_file_count += 1

    return written_file_count, num_docs, total_words


def proc_clueweb_parallel(file_paths, num_procs, chunk_size):
    # Split the data into chunks
    print("Starting multiprocessing of clueweb with " + str(num_procs) + " processes and a chunksize of " + str(
        chunk_size))
    chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    print("Split file paths into " + str(len(chunks)) + " chunks.")

    # Map the file paths to the reading function using the pool
    with multiprocessing.Pool(num_procs) as pool:
        results = pool.map(read_json_file, chunks)

    # Close the pool to indicate that no more tasks will be submitted
    pool.close()

    # Wait for all the processes in the pool to finish
    pool.join()

    # Return the contents of the JSON files
    return results


if __name__ == '__main__':
    clueweb_dir = "/mnt/datasets/clueweb2022_large/txt/pt"

    first_level_dirs = []
    for fl_dir in os.listdir(clueweb_dir):
        first_level_dirs.append(clueweb_dir + "/" + fl_dir)

    second_level_dirs = []
    for fl_dir in first_level_dirs:
        for sl_dir in os.listdir(fl_dir):
            second_level_dirs.append(fl_dir + "/" + sl_dir)

    print("Sub-directories:")
    print(len(second_level_dirs))

    file_paths = []
    for subdir in second_level_dirs:
        files = []
        for file in os.listdir(subdir):
            if file.endswith('.json.gz'):
                file_paths.append(subdir + "/" + file)

    print("Number of found files: " + str(len(file_paths)))

    num_procs = 12  # multiprocessing.cpu_count()

    # Debug
    # file_paths = file_paths[:32]
    # print("file paths len", len(file_paths))
    random.shuffle(file_paths)
    chunk_size = len(file_paths) // (num_procs * 4)
    start_time = datetime.now()
    # Read the JSON files using multiprocessing
    stats = proc_clueweb_parallel(file_paths, num_procs=num_procs, chunk_size=chunk_size)
    print(stats)
    end_time = datetime.now()
    written_file_count = 0
    num_docs = 0
    total_words = 0
    for s in stats:
        written_file_count += s[0]
        num_docs += s[1]
        total_words += s[2]

    print('Duration: {}'.format(end_time - start_time))
    print("Total docs: " + str(num_docs))
    print("Average words p/ doc for train split: " + str(total_words // num_docs))
    print("Total json files written: " + str(written_file_count))
