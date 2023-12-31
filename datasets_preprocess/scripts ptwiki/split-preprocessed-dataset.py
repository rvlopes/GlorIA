import argparse
import os

import jsonlines

# used to split ptwiki preprocess dataset into files
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="dataset", help='Dataset to use')
    parser.add_argument('-ds', dest='dataset_save', default="dataset", help='Dataset new name')
    parser.add_argument('-es', dest='eval_samples', default=10000, type=int, help='Num of eval samples for eval split')
    parser.add_argument('-ml', dest='max_length', default=512, type=int, help='Max document size in tokens/words')
    args = parser.parse_args()

    dataset_dir = "/data/rv.lopes/datasets/" + args.dataset
    dataset_file_paths = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.json'):
            dataset_file_paths.append(dataset_dir + "/" + file)
    print(dataset_file_paths)

    train_save_dir = "/data/rv.lopes/datasets/" + args.dataset_save + "-train"
    eval_save_dir = "/data/rv.lopes/datasets/" + args.dataset_save + "-eval"
    eval_samples = args.eval_samples

    docs = []
    docs_per_file = 100000
    total_train_count = 0
    total_eval_count = 0
    doc_count = 0
    filename = "doc_"
    train_file_dir = train_save_dir + "/" + filename
    eval_file_dir = eval_save_dir + "/" + filename
    curr_doc = ''
    num_words = []
    docs_analyzed = 0
    for f in dataset_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                curr_obj_len = len(obj['text'].split())
                if curr_obj_len > 0:
                    docs_analyzed = docs_analyzed + 1
                    curr_doc_len = len(obj['text'].split())
                    # if curr_obj_len + curr_doc_len > args.max_length:
                    if total_eval_count < eval_samples:
                        # Build eval docs
                        newDoc = {"id": total_eval_count, "text": obj['text'].strip()}
                        docs.append(newDoc)
                        total_eval_count = total_eval_count + 1
                        # curr_doc = obj['text']
                        if total_eval_count >= eval_samples:
                            with jsonlines.open(eval_file_dir + str(doc_count) + ".json", 'a') as d:
                                d.write_all(docs)
                            docs = []
                    else:
                        # Build train docs
                        newDoc = {"id": total_train_count, "text": obj['text'].strip()}
                        docs.append(newDoc)
                        total_train_count = total_train_count + 1
                        # curr_doc = obj['text']
                        num_words.append(curr_doc_len)
                        if total_train_count % docs_per_file == 0:
                            with jsonlines.open(train_file_dir + str(doc_count) + ".json", 'a') as d:
                                d.write_all(docs)
                            docs = []
                            doc_count = doc_count + 1
                    # else:
                    # else we append the documents and keep going
                    #    curr_doc = curr_doc + " ." + obj['text']

    # Write missing documents
    total_train_count = total_train_count + len(docs)  # add count of missing documents
    with jsonlines.open(train_file_dir + str(doc_count) + ".json", 'a') as d:
        d.write_all(docs)

    print("Total train examples: " + str(total_train_count))
    print("Total eval examples: " + str(total_eval_count))
    avg_words = sum(num_words) / total_train_count
    print("Average words p/ example: " + str(avg_words))
    print("Total docs analyzed: " + str(docs_analyzed))
"""
    train_save_dir = "/data/rv.lopes/datasets/" + args.dataset_save
    filename = "doc_"
    file_dir = train_save_dir + "/" + filename
    for f in dataset_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                newDoc = {"id": total_count, "text": obj['text']}
                docs.append(newDoc)
                total_count = total_count + 1
                if total_count % docs_per_file == 0:
                    with jsonlines.open(file_dir+str(doc_count) + ".json", 'a') as d:
                        d.write_all(docs)
                    docs = []
                    doc_count = doc_count + 1
"""
