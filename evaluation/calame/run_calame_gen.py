import csv
import tiktoken
from dotenv import load_dotenv
import os
import openai
import re
import jsonlines
import random
import itertools

# Load environment
load_dotenv("openai.env")

# Setup OpenAI API Auth
openai.api_key = os.getenv("OPENAI_API_KEY")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# Setup load dataset
def load_dataset(datasetName):
    dataset = []
    original_doc_count = 0
    dataset_file_paths = []
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


# Setup OpenAI prompt template
def setup_prompt(doc):
    return "Dado o seguinte contexto: " + doc + "\n\nEscreve um pequeno texto inspirado pelo contexto com poucas frases. Não deves mencionar nomes de pessoas ou países, eventos, marcas e datas (dias, anos e horas)."


# Setup call to OpenAI API
def request_generation(doc, temp=1, max_tokens=512, max_retries=3):
    msg_max_tokens = 2500
    msg = {
        "role": "user",
        "content": setup_prompt(doc['text'])
    }
    msg_tokens = num_tokens_from_messages([msg])
    res = None
    if msg_tokens < msg_max_tokens:
        for attempt in range(max_retries):
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        msg
                    ],
                    temperature=temp,
                    max_tokens=max_tokens
                )
                return res
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                print("Retrying...Attempt ", attempt)
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                print("Retrying...Attempt ", attempt)
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                print("Returning no response...")
                return None
        return res


def contains_only_special_characters(input_string):
    if re.match(r'^[^a-zA-Z0-9\s]+$', input_string):
        return True
    else:
        return False


def write_to_files(generated_data):
    print("Writing next 10 generations. Last processed doc is: ", proc_docs)
    with jsonlines.open(save_dir, 'a') as d:
        d.write_all(generated_data)



# Prepare new collection
generated_data = []
total_samples = 0
max_samples = 2500
min_words_in_org_doc = 30
max_words_in_org_doc = 600

calame_v = "calame"
print("Generation for ", calame_v)
save_dir = "/data/rv.lopes/datasets/"+calame_v+"/"+calame_v+".json"

# Load documents
print("Loading original datasets and preparing original collection...")
d1 = load_dataset("ptwikidocs-eval-clean")
d2 = load_dataset("arquivodocs-eval-clean")
d3 = load_dataset("oscar-ptpt-eval-clean")
random.seed(42)
random.shuffle(d1)
random.shuffle(d2)
random.shuffle(d3)

final_dataset = d1 + d2 + d3
random.shuffle(final_dataset)
print("Original collection prepared with a total of " + str(len(final_dataset)) + " documents.")

# Resume hardcoded
starting_doc = 0 #3138
total_samples = 0 #2320
proc_docs = 0 if starting_doc == 0 else starting_doc
if starting_doc > 0:
    print("Skipping to doc ", starting_doc)
    final_dataset = itertools.islice(final_dataset, starting_doc, None)


# Process documents - call OpenAI API to generate samples
print("Starting document generation pipeline...")
for doc in final_dataset:
    print("Processing document ", proc_docs)
    proc_docs += 1
    if total_samples < max_samples:
        words_in_doc = len(doc['text'].split())
        if min_words_in_org_doc <= words_in_doc <= max_words_in_org_doc:
            response = request_generation(doc)
            if response is not None:
                response_text = response["choices"][0]["message"]["content"]

                # Split multiple generations - 1 every new line
                gen_last_word_pairs = []
                generations = re.split('\n\n|\n', response_text)  # response_text.split("\n\n").split("\n")

                # We limit to 3 generations per document to avoid too much noise or repetition
                if len(generations) > 3:
                    generations = generations[:3]

                for gen in generations:
                    gen.replace("- ", "")
                    # Removes "1.", "2.", etc from the beginning of the samples
                    remove_numb = r'^\s*\d+[.)]\s*'
                    no_numbers = re.sub(remove_numb, '', gen, flags=re.MULTILINE)
                    # Get words of current generation
                    words = no_numbers.split(" ")
                    # Register last word / target word / word to be guessed
                    last_word = words[-1].replace(".", "")
                    # Skip generations whose last word is none, only numbers or only special characters
                    # since it doesnt make sense for models to guess them
                    drop_generation = last_word == "" or contains_only_special_characters(
                        last_word) or last_word.isdigit()
                    if drop_generation:
                        break
                    # Register sentence/sample without last word
                    modified_sentence = " ".join(words[:-1])
                    modified_sentence = modified_sentence.replace("- ", "")
                    # Create sentence (no last word) + last_word pair and add to collection
                    gen_pair = {"id": total_samples, "sentence": modified_sentence, "last_word": last_word}
                    print("PAIR", gen_pair)
                    total_samples += 1
                    generated_data.append(gen_pair)
                    if total_samples % 10 == 0:
                        write_to_files(generated_data)
                        generated_data = []
    else:
        break

print("Finished generation. Writing remaining data to .json...")
# Write to disk
write_to_files(generated_data)

print("Processed documents: ", proc_docs)
print("Total samples:", total_samples)

# Load generated dataset and write it to csv for review
calame = load_dataset(calame_v)
print("Writing dataset to .csv for review...")
# Write data to CSV
with open(calame_v+"_csv.csv", mode='a', newline='', encoding='utf-8') as file:
    fieldnames = ['id', 'sentence', 'last_word']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()  # Write header row

    for item in calame:
        writer.writerow(item)  # Write each item as a row


