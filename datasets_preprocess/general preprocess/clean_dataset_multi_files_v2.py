import argparse
import os
import string
from datetime import datetime
import jsonlines
import numpy as np
import sentencepiece
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from simhash import Simhash, SimhashIndex
import ftfy
from ftfy import fix_text
import re
from datasets import load_dataset
import emoji
import kenlm

# used by every dataset except clueweb due to it having another dedicated script


# Misc setups for characters
main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
emoji = list(emoji.UNICODE_EMOJI["en"].keys())
special_characters_default = set(main_special_characters + other_special_characters)
special_characters_default.update(emoji)
whitespaces_list = [
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
    " ",
    " ",
    " ",
    " ",
    "￼",
    "",
]
# Paramters for filtering

parameters_filtering_pt = {
    "cond_uniform_whitespace": True,
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 19,
    "cond_check_number_words": True,
    "tokenization": False,
    "strip_characters": special_characters_default,
    "number_words_min_cutoff": 19,
    "number_words_max_cutoff": 100000,
    "cond_check_character_repetition_removal": True,
    "character_repetition_length": 10,
    "character_repetition_max_cutoff": 0.25,
    "cond_check_word_repetition_removal": True,
    "word_repetition_length": 5,
    "word_repetition_max_cutoff": 0.98,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.35,
    "cond_words_augmentation": False,
    "words_augmentation_group_sizes": [],
    "words_augmentation_join_char": "",
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.2,
    "cond_check_flagged_words": True,
    "flagged_words_max_cutoff": 0.007,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.6,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 3038,
}


# Load models for PPL
def load_sentencepiece_model():
    path_sentencepiece_model = "/data/rv.lopes/models/bloompreprocess/pt.sp.model"
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    sentencepiece_model.load(path_sentencepiece_model)
    return sentencepiece_model


def load_kenlm_model():
    path_kenlm_model = "/data/rv.lopes/models/bloompreprocess/pt.arpa.bin"
    kenlm_model = kenlm.Model(path_kenlm_model)
    return kenlm_model


sentencepiece_m = load_sentencepiece_model()
kenlm_m = load_kenlm_model()


def load_dataset_hf(datasetName, pre_split=False):
    dataset_dir = "/data/rv.lopes/datasets/" + datasetName
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

    dataset = load_dataset("json", data_files=dataset_file_paths, split="train")
    return dataset


def remove_empty_el_from_list(list_):
    return [el for el in list_ if el]


def split_on_whitespace(
        document,
        new_line=False,
        tab=False,
):
    """This method also removes concatenated spaces."""
    sep = [" "] + new_line * ["\n"] + tab * ["\t"]
    sep = "|".join(sep)
    split_document = re.split(sep, document)
    split_document = remove_empty_el_from_list(split_document)
    return split_document


def split_on_newline_tab_whitespace(document):
    """First split on "\n", then on "\t", then on " "."""
    sentences = document.split("\n")
    sentences = [sentence.split("\t") for sentence in sentences]
    sentences = [
        [
            split_on_whitespace(subsentence)
            for subsentence in sentence
        ]
        for sentence in sentences
    ]
    return sentences


def uniform_whitespaces(document):
    whitespace = set(whitespaces_list)
    document = "".join(
        [char if char not in whitespace else " " for char in document]
    )
    return document


def strip(document, strip_characters):
    """Way faster than document.strip(strip_characters)
        since strip_characters is now a set instead of a str,
        and it contains a lot of elements (all the emojis)."""
    if not document:
        return document
    beg_ind = 0
    end_ind = len(document)
    for i in range(len(document)):
        if document[i] in strip_characters:
            beg_ind += 1
        else:
            break
    for i in range(1, len(document) + 1):
        if document[-i] in strip_characters:
            end_ind -= 1
        else:
            break
    document_stripped = document[beg_ind:end_ind]
    return document_stripped


def should_keep_word_with_incorrect_substrings(
        word, strip_characters, incorrect_word_substrings
):
    word = strip(word, strip_characters)
    should_keep = all(
        [(i_substr not in word) for i_substr in incorrect_word_substrings]
    )
    return should_keep


def merge_on_whitespace_tab_newline(sentences):
    """Invert the method split_on_newline_tab_whitespace.
        Removes concatenated separators."""
    sentences = [
        [" ".join(subsentence) for subsentence in sentence if subsentence]
        for sentence in sentences
    ]
    sentences = ["\t".join(sentence) for sentence in sentences if sentence]
    if not sentences:
        return ""
    document = "\n".join(sentences)
    return document


def remove_incorrect_substrings(document):
    incorrect_word_substrings = parameters_filtering_pt['incorrect_word_substrings']
    strip_characters = special_characters_default
    sentences = split_on_newline_tab_whitespace(document)
    sentences = [
        [
            [
                word
                for word in subsentence
                if should_keep_word_with_incorrect_substrings(
                word, strip_characters, incorrect_word_substrings
            )
            ]
            for subsentence in sentence
        ]
        for sentence in sentences
    ]
    document = merge_on_whitespace_tab_newline(sentences)
    return document


def modify_doc_map(document):
    document['text'] = modify_doc(document['text'])
    return document


def modify_doc(document_text):
    # Strip
    document_text = document_text.strip()
    # Uniform whitespaces
    document_text = uniform_whitespaces(document_text)
    # Remove incorrect word substrings
    document_text = remove_incorrect_substrings(document_text)
    return document_text


# OTHER
def remove_pub_prefix(document):
    # Define the regex pattern to match "Pub" at the start of the string
    pattern = r'^Pub'
    # Use the sub() function from the re module to replace the matched pattern with an empty string
    result = re.sub(pattern, '', document)
    return result


def remove_duplicates_set(dataset):
    seen_texts = set()
    for dc in dataset:
        if dc['text'] not in seen_texts:
            seen_texts.add(dc['text'])

    return list(seen_texts)


def compute_character_repetition_ratio(document, character_repetition_length):
    def get_freq_character_ngrams(document, n):
        character_ngrams = [
            document[i: i + n] for i in range(len(document) - n + 1)
        ]
        freq_character_ngrams = {}
        for character_ngram in character_ngrams:
            freq_character_ngrams[character_ngram] = (
                    freq_character_ngrams.get(character_ngram, 0) + 1
            )
        return freq_character_ngrams

    freq_character_ngrams = get_freq_character_ngrams(
        document, character_repetition_length
    )
    if len(freq_character_ngrams) == 0:
        return 0
    freq_character_ngrams = list(freq_character_ngrams.values())
    freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
    val_one = len([el for el in freq_character_ngrams if el == 1])
    num_rep_character_ngrams = min(
        int(np.sqrt(len(freq_character_ngrams))),
        len(freq_character_ngrams) - val_one,
    )
    character_repetition_ratio = sum(
        freq_character_ngrams[:num_rep_character_ngrams]
    ) / sum(freq_character_ngrams)
    return character_repetition_ratio


def filter_doc_by_character_repetition(document):
    character_repetition_ratio = compute_character_repetition_ratio(
        document, parameters_filtering_pt["character_repetition_length"]
    )
    cond = character_repetition_ratio <= parameters_filtering_pt["character_repetition_max_cutoff"]
    return cond


def compute_word_repetition_ratio(
        document, sentencepiece_model_tok, strip_characters, word_repetition_length
):
    def get_freq_word_ngrams(
            document, sentencepiece_model_tok, strip_characters, n
    ):
        words = document.split()
        word_ngrams = [
            " ".join(words[i: i + n]) for i in range(len(words) - n + 1)
        ]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1
        return freq_word_ngrams

    freq_word_ngrams = get_freq_word_ngrams(
        document, sentencepiece_model_tok, strip_characters, word_repetition_length
    )
    if len(freq_word_ngrams) == 0:
        return 0
    freq_word_ngrams = list(freq_word_ngrams.values())
    word_repetition_ratio = sum(
        freq for freq in freq_word_ngrams if freq > 1
    ) / sum(freq_word_ngrams)
    return word_repetition_ratio


def filter_doc_by_word_repetition(document):
    word_repetition_ratio = compute_word_repetition_ratio(
        document, sentencepiece_m,  # parameters_filtering_pt["sentencepiece_model_tok"],
        parameters_filtering_pt["strip_characters"],
        parameters_filtering_pt["word_repetition_length"]
    )
    cond = word_repetition_ratio <= parameters_filtering_pt["word_repetition_max_cutoff"]
    return cond


def compute_perplexity_score(document, sentencepiece_model, kenlm_model):
    document = modify_doc(document)
    document = sentencepiece_model.encode_as_pieces(document)
    document = " ".join(document)
    doc_log_score, doc_length = 0, 0
    for line in document.split("\n"):
        log_score = kenlm_model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
    pp_score = 10.0 ** (-doc_log_score / doc_length)
    pp_score = round(pp_score, 1)
    return pp_score


def filter_by_ppl(document):
    cond = True
    score = compute_perplexity_score(
        document, sentencepiece_m, kenlm_m
    )
    cond = score <= parameters_filtering_pt["perplexity_max_cutoff"]
    return cond


def filter_docs(document):
    if not filter_doc_by_character_repetition(document['text']):
        return False
    if not filter_doc_by_word_repetition(document['text']):
        return False
    if not filter_by_ppl(document['text']):
        return False
    return True


def fix_mojibakes_map(document):
    document['text'] = remove_pub_prefix(document['text'])
    document['text'] = ftfy.fix_text(document['text'])
    return document


def remove_by_word_count(document):
    words = document['text'].split()
    if len(words) < min_words:
        return False
    return True


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
    min_words = 20

    # Load dataset
    print("Loading dataset " + datasetName)
    start_time = datetime.now()
    dataset = load_dataset_hf(datasetName, pre_split=False)
    dataset = dataset.remove_columns([column for column in dataset.column_names if column != "text"])
    # dataset = dataset[:20000]
    end_time = datetime.now()
    print('Load dataset finished. Duration: {}'.format(end_time - start_time))

    modify_docs = True
    do_mojibakes = True
    do_dup_removal = True
    do_remove_by_word_count = True

    # Modify and clean document content
    if modify_docs:
        print("Starting document cleaning/modification")
        start_time = datetime.now()
        dataset = dataset.map(modify_doc_map)
        end_time = datetime.now()
        print('Doc modification finished. Duration: {}'.format(end_time - start_time))

    print("Starting document filtering")
    start_time = datetime.now()
    dataset = dataset.filter(filter_docs)
    end_time = datetime.now()
    print('Doc filtering finished. Duration: {}'.format(end_time - start_time))

    # Fix mojibakes
    if do_mojibakes:
        print("Starting mojibakes cleanup")
        start_time = datetime.now()
        dataset = dataset.map(fix_mojibakes_map)
        end_time = datetime.now()
        print('Fix mojibakes finished. Duration: {}'.format(end_time - start_time))

    if do_remove_by_word_count:
        # Remove docs by word count
        print("Starting docs removal by word count")
        start_time = datetime.now()
        dataset = dataset.filter(remove_by_word_count)
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
    train_save_dir = "/data/rv.lopes/datasets/" + datasetSave
    file_prefix = "doc_"
    train_file_dir = train_save_dir + "/" + file_prefix
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
        a_d = doc if do_dup_removal else doc['text']
        num_words = len(a_d.split())
        total_num_words += num_words
        # prepare doc to add to buffer
        newDoc = {"id": total_train_count, "text": a_d}
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
