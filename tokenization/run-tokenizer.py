import os
import json
import sys
from pathlib import Path
import pprint as pp
import argparse
from transformers import AutoTokenizer
from tokenizer_trainer import TokenizerTrainer

if __name__ == '__main__':
    # Read config json
    f = open("tokenizer-config.json")
    config = json.load(f)

    parser = argparse.ArgumentParser(description='Dataset preprocessing script')
    parser.add_argument('-c', dest='configName', default="BertConfig", help='Config to use')
    parser.add_argument('-t', dest='tokenizerName', default="mariaalBERTina-tokenizer-v0", help='Tokenizer name')

    # Parse arguments
    args = parser.parse_args()

    # Process Configuration
    configName = args.configName
    # modelName = config["ModelName"]
    config = config['Configs'][configName]
    pp.pprint("##### CONFIG #####")
    pp.pprint(config)

    # Pathing - manually add datasets path
    ptwiki_dir = "/data/rv.lopes/datasets/ptwikidocs-train-clean"
    arquivo_dir = "/data/rv.lopes/datasets/arquivodocs-train-clean"
    europarl_dir = "/data/rv.lopes/datasets/europarldocs-train"
    datasets_dir = [ptwiki_dir, arquivo_dir, europarl_dir]
    dataset_file_paths = []
    for dataset in datasets_dir:
        for file in os.listdir(dataset):
            if file.endswith('.json'):
                dataset_file_paths.append(dataset + "/" + file)
        print(dataset_file_paths)

    # Tokenizer path
    tokenizerName = args.tokenizerName
    tokenizerPathing = "/data/rv.lopes/tokenizers"
    tokenizerPath = os.path.join(tokenizerPathing, tokenizerName)
    pp.pprint("Tokenizer path: " + tokenizerPath)

    # Load tokenizer trainer
    pp.pprint("Creating tokenizer trainer")
    tokenTrainer = TokenizerTrainer(config, dataset_file_paths, tokenizerPath)
    pp.pprint("Training tokenizer")
    tokenTrainer.train()
    pp.pprint("Saving tokenizer")
    tokenTrainer.save()

    # Quick tokenizer test
    print("TESTING TOKENIZER")
    testString = "A América Latina (; ) é uma região do continente americano que engloba os países onde são faladas, primordialmente, línguas românicas (derivadas do latim) — no caso, o espanhol, o português e o francês — visto que, historicamente, a região foi maioritariamente dominada pelos impérios coloniais europeus Espanhol e Português. A América Latina tem uma área de cerca de km², o equivalente a cerca de 3,9% da superfície da Terra (ou 14,1% de sua superfície emersa terrestre)."
    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizerPath, max_len=config['maxLength'])
    encoding = loaded_tokenizer.encode(testString)
    print(encoding)
    print(loaded_tokenizer.decode(encoding))
