import argparse
import os

import jsonlines
# used to group texts in arquivo docs and split it into files
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="ptwikicleanv2", help='Dataset to use')
    parser.add_argument('-ds', dest='dataset_save', default="ptwikicleanv2", help='Where to save')
    parser.add_argument('-ml', dest='max_length', default=512, type=int, help='Max document size in tokens/words')

    args = parser.parse_args()

    dataset_dir = "/data/rv.lopes/datasets/" + args.dataset
    dataset_file_paths = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.json'):
            dataset_file_paths.append(dataset_dir + "/" + file)
    print(dataset_file_paths)

    # Vars to split dataset into files
    docs = []
    docs_per_file = 100000
    docs_to_write = 0
    total_count = 0
    doc_count = 0
    save_dir = "/data/rv.lopes/datasets/" + args.dataset_save
    filename = "doc_"
    file_dir = save_dir + "/" + filename
    curr_doc = ''
    for f in dataset_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                # obj['id'] = total_count
                curr_obj_len = len(obj['text'].split())
                if curr_obj_len > 0:  # sanity check for empty text - should not happen - skip them
                    curr_doc_len = len(curr_doc.split())
                    if curr_obj_len + curr_doc_len > args.max_length:
                        # If the current iterated document and current doc go over the limit, add only current doc
                        newDoc = {"id": total_count, "text": curr_doc.strip()}
                        docs.append(newDoc)
                        total_count = total_count + 1
                        curr_doc = obj['text']
                        if total_count % docs_per_file == 0:
                            with jsonlines.open(file_dir + str(doc_count) + ".json", 'a') as d:
                                d.write_all(docs)
                            docs = []
                            doc_count = doc_count + 1
                    else:
                        # else we append the documents and keep going
                        curr_doc = curr_doc + " ." + obj['text']

    docs.append({"text": curr_doc.strip(), "id": total_count})
    # Write missing documents
    with jsonlines.open(file_dir + str(doc_count) + ".json", 'a') as d:
        d.write_all(docs)
