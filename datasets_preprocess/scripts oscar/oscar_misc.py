from huggingface_hub import login
from datasets import load_dataset
from jsonlines import jsonlines

# Script to download oscar into multiple json docs to be used in our other clean dataset scripts
if __name__ == '__main__':
    token = "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"
    login(token=token)

    dataset = load_dataset("oscar-corpus/OSCAR-2201",
                           use_auth_token=True,  # required
                           language="pt",
                           streaming=True,  # optional
                           split="train")  # optional, but the dataset only has a train split

    docs_to_process = 2
    docs_processed = 0
    ptpt_docs = 0
    docs = []
    for doc in dataset:
        url = doc['meta']['warc_headers']['warc-target-uri']
        if ".pt" in url:
            new_doc = {"id": ptpt_docs, "text": doc['text']}
            docs.append(new_doc)
            ptpt_docs += 1
    print("Found " + str(ptpt_docs) + " documents for the .pt domain.")

    # Dataset split directories
    datasetSave = "oscar-ptpt"
    save_dir = "/user/home/rv.lopes/data/datasets/" + datasetSave
    file_prefix = "doc_"
    train_file_dir = save_dir + "/" + file_prefix
    # Prep some counters :)
    docs_per_file = 100000  # max number of documents per json file (in train split)
    file_count = 0  # number of written files
    doc_count = 0
    # Prep documents to write
    doc_buffer = []

    for doc in docs:
        doc_buffer.append(doc)
        doc_count += 1
        if doc_count % docs_per_file == 0:
            with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
                d.write_all(doc_buffer)
            doc_buffer = []
            file_count += 1

    with jsonlines.open(train_file_dir + str(file_count) + ".json", 'a') as d:
        d.write_all(doc_buffer)
