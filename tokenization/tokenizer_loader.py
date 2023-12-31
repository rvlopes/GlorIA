import os
from transformers import BertTokenizer, AutoTokenizer, GPT2TokenizerFast


# Loads already trained tokenizers
class TokenizerLoader:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizersDir = "/data/rv.lopes/tokenizers" #"./mariaalBERTina"

    def loadTokenizer(self, maxLength, baseModel):
        tokenizerPath = self.tokenizersDir + "/" + self.tokenizer
        if "BERT" in baseModel:
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizerPath, max_len=maxLength)
        if "GPT2" in baseModel or "NEO" in baseModel or "BLOOM" in baseModel:
            loaded_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizerPath, max_len=maxLength)
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
        print(f"##### Tokenizer {self.tokenizer} loaded. #####")
        return loaded_tokenizer
