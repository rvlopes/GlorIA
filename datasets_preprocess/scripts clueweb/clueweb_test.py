import gzip
import json
import os
import time
from datetime import datetime
import ftfy
import jsonlines

if __name__ == '__main__':
    # clueweb_dir = "/mnt/nas/datasets/clueweb2022/txt/pt/pt00"
    clueweb_dir = "/mnt/datasets/clueweb2022_large/txt/pt"

    train_save_dir = "/data/rv.lopes/datasets/clueweb-ptpt-train/"
    eval_save_dir = "/data/rv.lopes/datasets/clueweb-ptpt-eval/"

    # /pt0000/
    first_level_dirs = []
    for fl_dir in os.listdir(clueweb_dir):
        first_level_dirs.append(clueweb_dir + "/" + fl_dir)

    second_level_dirs = []
    for fl_dir in first_level_dirs:
        for sl_dir in os.listdir(fl_dir):
            second_level_dirs.append(fl_dir + "/" + sl_dir)

    print("Sub-directories:")
    print(len(second_level_dirs))

    docs = []
    doc_count = 0
    total_words = 0
    start_time = datetime.now()
    docs_per_file = 100000
    eval_docs = 10000
    total_train_count = 0
    total_eval_count = 0
    for subdir in second_level_dirs:
        files = []
        for file in os.listdir(subdir):
            if file.endswith('.json.gz'):
                files.append(subdir + "/" + file)

        print("Found " + str(len(files)) + " files")
        print(files)

        for f in files:
            with gzip.open(f, 'r') as reader:
                j_reader = jsonlines.Reader(reader)

                for doc in j_reader:
                    if ".pt" in doc['URL']:
                        doc_words = len(doc['Clean-Text'].split())
                        total_train_count += 1
                        """
                        if doc_words >= 15:
                            if total_eval_count < eval_docs:
                                cleaned_text = ftfy.fix_text(doc['Clean-Text'].strip())
                                newDoc = {"id": total_eval_count, "text": cleaned_text}
                                docs.append(newDoc)
                                total_eval_count += 1
                                if total_eval_count >= eval_docs:
                                    with jsonlines.open(eval_save_dir + "doc_" + str(doc_count) + ".json", 'a') as d:
                                        d.write_all(docs)
                                    docs = []
                            else:
                                cleaned_text = ftfy.fix_text(doc['Clean-Text'].strip())
                                # Word stats for train split
                                total_words += doc_words
                                # Add train entry
                                newDoc = {"id": total_train_count, "text": cleaned_text}
                                docs.append(newDoc)
                                total_train_count += 1
                                if total_train_count % docs_per_file == 0:
                                    with jsonlines.open(train_save_dir + "doc_" + str(doc_count) + ".json", 'a') as d:
                                        d.write_all(docs)
                                    docs = []
                                    doc_count += 1
                        
    # Write missing documents
    total_train_count = total_train_count + len(docs)  # add count of missing documents
    with jsonlines.open(train_save_dir + str(doc_count) + ".json", 'a') as d:
        d.write_all(docs)
    """

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print("Total docs: " + str(doc_count))
    print("Average words p/ doc for train split: " + str(total_words // total_train_count))
    print("Total train examples: " + str(total_train_count))
    print("Total eval examples: " + str(total_eval_count))

