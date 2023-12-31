import os
from pathlib import Path

import jsonlines
import re

# Script loads scraped texts and stores them in a cleaned format with
# title and text only
if __name__ == '__main__':

    # Using cluster's dirs
    # 500 sample - "/user/data/arquivopt/scraped_texts/500_sample" + "/" + "scraped_sample500.jsonl"
    # fulljan - "/user/data/arquivopt/scraped_texts/26Jan2023"  + "/" + "scraped_full_26Jan2023.jsonl"
    scrapeDir = "/user/data/arquivopt/scraped_texts/500_sample"
    targetScrape = scrapeDir + "/" + "scraped_sample500.jsonl"

    # permission denied: "/user/data/arquivopt/scraped_texts/500_sample_clean"
    # fulljan - "/data/rv.lopes/datasets/pt_web_corpus/26Jan2023" + "/cleaned_full_26Jan2023.json"
    # 500 sample - "/data/rv.lopes/datasets/pt_web_corpus/500_sample_clean" + "/cleaned_sample500.json"
    saveDir = "/data/rv.lopes/datasets/pt_web_corpus/500_sample_clean"
    saveFile = saveDir + "/cleaned_sample500.json"

    # Create save folder - sanity check!!
    if not os.path.exists(saveDir) or not os.path.isdir(saveDir):
        Path(saveDir).mkdir(exist_ok=True)

    entries = []
    num_tokens = 0
    avg_tokens = 0
    total_entries = 0
    write_every = 2000
    with jsonlines.open(targetScrape) as reader:
        for obj in reader:
            total_entries = total_entries + 1
            cleaned_obj = {"title": obj['title'],
                           "text": obj['body']}
            text_tokens = len(re.findall(r'\w+', obj['body']))
            num_tokens = num_tokens + text_tokens
            entries.append(cleaned_obj)
            if total_entries % write_every == 0:
                print("Writing 2000...")
                with jsonlines.open(saveFile, mode='a') as writer:
                    #writer.write(entries)
                    writer.write_all(entries)
                    entries = []

    # Append remainders

    #with jsonlines.open(saveFile, mode='a') as writer:
    #    writer.write(entries)
    print("Save file: "+ saveFile)
    with jsonlines.open(saveFile, 'a') as f:
        f.write_all(entries)

    print("Total entries processed: " + str(total_entries))
    print("Total tokens (estimated): " + str(num_tokens))
    print("Average tokens p/ entry (in body): " + str(num_tokens / total_entries))
