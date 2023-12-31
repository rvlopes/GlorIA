from datetime import datetime
import multiprocessing
import os
import jsonlines

from transformers import GPT2TokenizerFast

# Load our tokenizer
tokenizers_dir = "/data/rv.lopes/tokenizers/"
loaded_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizers_dir + "gptuga-tk-512")


def read_json_file(file_paths):
    total_tokens = 0
    for f in file_paths:
        with jsonlines.open(f) as reader:
            for doc in reader:
                doc_tokens = len(loaded_tokenizer(doc['text']).input_ids)
                total_tokens += doc_tokens

    return total_tokens


def proc_dataset_tokenization_parallel(dataset, file_paths, num_procs, chunk_size):
    # Split the data into chunks
    print("Starting multiprocessing of " + dataset + " with " + str(num_procs) + " processes and a chunksize of " + str(
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
    # Prepare list of datasets
    datasets_folder = "/data/rv.lopes/datasets"
    datasets = ["ptwikidocs-train-clean", "arquivodocs-train-clean",
                "europarldocs-train", "oscar-ptpt-train-clean",
                "opensubtitles-filtered"] #clueweb done separately

    for d in datasets:
        # Build dataset dir
        cur_dataset_dir = datasets_folder + "/" + d
        print("Starting token count for dataset: ", d)
        # Get dataset json files
        dataset_file_paths = []
        for file in os.listdir(cur_dataset_dir):
            if file.endswith('.json'):
                dataset_file_paths.append(cur_dataset_dir + "/" + file)
        print("Found files", dataset_file_paths)

        total_tokens = 0
        num_procs = 12
        num_files_found = len(dataset_file_paths)
        if num_procs >= num_files_found:
            num_procs = num_files_found
        chunk_size = num_files_found // num_procs #(num_procs * 4)
        print(num_files_found, num_procs, chunk_size)
        start_time = datetime.now()
        results = proc_dataset_tokenization_parallel(d, dataset_file_paths,
                                                     num_procs=num_procs,
                                                     chunk_size=chunk_size)
        for res in results:
            total_tokens += res
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        print("Total token count: ", total_tokens)
        print("###########################################")
