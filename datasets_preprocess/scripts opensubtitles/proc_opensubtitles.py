import jsonlines
from datetime import datetime
import re
from bad_words import get_bad_words

def filter_bad_words(sentence, bad_words):
    pattern = r"\b(" + "|".join(bad_words) + r")\b"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    return bool(match)

if __name__ == '__main__':
    # opensub_dir = "/data/dsemedo/opensubtitles.raw.pt.text"
    # /user/home/rv.lopes/data/
    opensub_dir = "/user/home/rv.lopes/data/datasets/opensub_org/opensubtitles.raw.pt.txt"
    train_dir = "/user/home/rv.lopes/data/datasets/opensubtitles-filtered/"

    bad_words = get_bad_words()
    docs = []
    num_docs = 0
    docs_per_file = 100000
    written_files = 0
    max_words_p_doc = 500
    total_words = 0
    start_time = datetime.now()
    with open(opensub_dir, 'r') as file:
        concatenated_lines = ''
        words_count = 0
        # Read and process each line
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if not filter_bad_words(line, bad_words):
                line_word_count = len(line.split())
                total_words += line_word_count
                # Check if concatenating the line exceeds the word limit
                if words_count + line_word_count <= max_words_p_doc:
                    concatenated_lines += line + '\n '
                    words_count += line_word_count
                else:
                    # Perform processing on the concatenated lines (replace this with your own logic)
                    newDoc = {"id": num_docs, "text": concatenated_lines}
                    docs.append(newDoc)
                    num_docs += 1
                    if num_docs % docs_per_file == 0:
                        with jsonlines.open(train_dir + "doc_" + str(written_files) + ".json", 'a') as d:
                            d.write_all(docs)
                        docs = []
                        written_files += 1

                    # Reset variables for the next set of lines
                    concatenated_lines = line + '\n '
                    words_count = line_word_count

        newDoc = {"id": num_docs, "text": concatenated_lines}
        docs.append(newDoc)
        num_docs += 1
        with jsonlines.open(train_dir + "doc_" + str(written_files) + ".json", 'a') as d:
            d.write_all(docs)
        docs = []
        written_files += 1

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print("Total docs: " + str(num_docs))
    print("Average words p/ doc for train split: " + str(total_words // num_docs))