import argparse
from dataset_loader import DatasetLoader
import re
import pprint as pp
import spacy
import json
from tqdm import tqdm
import jsonlines


def cjk_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    return None


# 1.6M / 32 = 50k p/Epoch
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-d', dest='dataset', default="ptwiki", help='Dataset to use')

    # Parse arguments
    args = parser.parse_args()

    # Spacy
    nlp = spacy.load('pt_core_news_sm')

    # Directories and paths
    cleanDatasetName = args.dataset + "clean"
    datasetsFolder = "./datasets"
    datasetDir = datasetsFolder + "/" + cleanDatasetName
    datasetFile = datasetDir + "/" + cleanDatasetName + ".json"

    # Create dataset loader and splitter
    datasetLoader = DatasetLoader(args.dataset)
    # Load dataset
    datasetLoader.loadDataset()
    # Get dataset splits
    splitDataset = datasetLoader.getSplits(printSplits=True)

    maxLength = 512

    entries = []

    test = 0

    # Process dataset entries - TODO
    for entry in tqdm(splitDataset['train']):
        removedSentences = []
        if cjk_detect(entry['text']) is None:
            words = re.findall(r'\w+', entry['text'])
            totalWords = len(words)
            if totalWords > 0:
                if totalWords <= maxLength:
                    entries.append({
                        "title": entry['title'],
                        "text": entry['text']
                    })
                else:  # if larger than max length, we need to truncat and split into other entries
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', entry['text'])
                    # Remove sentence a first time
                    removedWords = 0
                    removedSentence = sentences.pop()
                    removedSentences.append(removedSentence)
                    wordsInSentence = len(re.findall(r'\w+', removedSentence))
                    removedWords = removedWords + wordsInSentence
                    while totalWords - removedWords > 512:
                        removedSentence = sentences.pop()
                        removedSentences.append(removedSentence)
                        wordsInSentence = len(re.findall(r'\w+', removedSentence))
                        removedWords = removedWords + wordsInSentence

                    text = ' '.join(sentences)
                    entries.append({
                        "title": entry['title'],
                        "text": text
                    })

                    currText = ""
                    lastIdx = 0
                    for idx, s in enumerate(removedSentences):
                        currWords = len(re.findall(r'\w+', currText))
                        toProcessWords = len(re.findall(r'\w+', s))
                        if idx == 0 or idx != lastIdx:
                            if currWords + toProcessWords <= maxLength:
                                currText = currText + " " + s
                                tempWords = len(re.findall(r'\w+', currText))
                                nextWords = 0
                                if idx < len(removedSentences) - 1:  # if has next
                                    nextWords = len(re.findall(r'\w+', removedSentences[idx + 1]))
                                    if tempWords + nextWords <= maxLength:
                                        currText = currText + " " + removedSentences[
                                            idx + 1]  # concat curr sentence with next sentence
                                        lastIdx = idx + 1
                                    else:
                                        entries.append({"title": entry['title'],
                                                        "text": currText})
                                        currText = ''
                                else:
                                    entries.append({"title": entry['title'],
                                                    "text": currText})
                                    currText = ''
                            else:
                                entries.append({"title": entry['title'],
                                                "text": currText})
                                if idx == len(removedSentences) - 1 and toProcessWords <= maxLength and idx != lastIdx:
                                    entries.append({"title": entry['title'],
                                                    "text": s})
                                currText = ''
        #test = test + 1
        #if test == 2:
        #    break
    with jsonlines.open(datasetFile, 'w') as f:
        f.write_all(entries)
