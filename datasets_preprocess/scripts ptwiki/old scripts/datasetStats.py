import argparse
from dataset_loader import DatasetLoader
import re
import pprint as pp
from tqdm import tqdm

# 1.6M / 32 = 50k p/Epoch
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset stats script')
    parser.add_argument('-d', dest='dataset', default="ptwikiclean", help='Dataset to use')

    # Parse arguments
    args = parser.parse_args()

    # Create datasets loader and splitter
    datasetLoader = DatasetLoader(args.dataset)
    # Load datasets
    datasetLoader.loadDataset()
    # Get datasets splits
    splitDataset = datasetLoader.getSplits(printSplits=True)

    maxLength = 512
    wordCount = 0
    sentencesCount = 0
    actualUsedWordCount = 0
    numRows = splitDataset.num_rows['train']
    asciiRows = 0
    nonAsciiRows = 0

    for text in tqdm(splitDataset['train']['text']):
        # Word counting
        words = re.findall(r'\w+', text)
        sentencesCount = sentencesCount + len(re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text))
        totalWords = len(words)
        wordCount = wordCount + totalWords
        if totalWords >= maxLength:
            actualUsedWordCount = actualUsedWordCount + maxLength
        else:
            actualUsedWordCount = actualUsedWordCount + totalWords

    pp.pprint("Total number of entries in datasets (text belonging to wiki pages): " + str(numRows))
    pp.pprint("Total Words/Tokens in training datasets: " + str(wordCount))
    pp.pprint("Total actual tokens used ( text.len <= 512): " + str(actualUsedWordCount))
    pp.pprint("Total number of sentences: " + str(sentencesCount))
