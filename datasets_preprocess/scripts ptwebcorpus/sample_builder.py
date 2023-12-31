import os
import random
from pathlib import Path

import jsonlines
import re

if __name__ == '__main__':

    ptwiki_samples = 128  # weight 0.25
    arquivo_samples = 384  # weight 0.75

    # PTWIKI
    ptwiki_dir = "/data/rv.lopes/datasets/ptwikicleanv2"
    ptwiki_file_paths = []
    for file in os.listdir(ptwiki_dir):
        if file.endswith('.json'):
            ptwiki_file_paths.append(ptwiki_dir + "/" + file)
    print(ptwiki_file_paths)

    # ARQUIVO
    arquivo_dir = "/data/rv.lopes/datasets/pt_web_corpus/500_sample_clean"
    arquivo_file_paths = []
    for file in os.listdir(arquivo_dir):
        if file.endswith('.json'):
            arquivo_file_paths.append(arquivo_dir + "/" + file)
    print(arquivo_file_paths)

    samples = []

    # read ptwiki files
    pt_s = 0
    for f in ptwiki_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                annotated_sample = {
                    "text": obj['text'],
                    "dataset": "PTWIKI"
                }
                samples.append(annotated_sample)
                pt_s = pt_s + 1
                if pt_s >= ptwiki_samples:
                    break

    # read arquivo files
    arq_s = 0
    for f in arquivo_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                if obj['text']:
                    annotated_sample = {
                        "text": obj['text'],
                        "dataset": "ARQUIVO"
                    }
                    samples.append(annotated_sample)
                    arq_s = arq_s + 1
                    if arq_s >= arquivo_samples:
                        break

    # Create save folder - sanity check!!
    mixed_sample_dir = "/data/rv.lopes/datasets/pt_web_corpus/mixed_sample"
    if not os.path.exists(mixed_sample_dir) or not os.path.isdir(mixed_sample_dir):
        Path(mixed_sample_dir).mkdir(exist_ok=True)

    random.shuffle(samples)
    mixed_sample_file = mixed_sample_dir + "/mixed_sample.json"

    with jsonlines.open(mixed_sample_file, 'a') as f:
        f.write_all(samples)
