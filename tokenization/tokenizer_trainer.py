from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers.models.bert import BertTokenizerFast


class TokenizerTrainer:
    tokenizer = None
    trainer = None
    file_paths = []
    tokenizerPath = None

    def __init__(self, config, file_paths, tokenizerPath):
        self.config = config
        self.tokenizerPath = tokenizerPath
        vocab_size = self.config['vocabSize']
        special_tokens = self.config['specTokens']

        # Paths of files to use for training
        self.file_paths = file_paths
        # Tokenizer type to use
        if self.config['tokenizerType'] == 'WordPiece':
            self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            self.tokenizer.decoder = decoders.WordPiece(prefix="##")
            self.trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
            # Type of pre tokenization we want
            self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

            # Padding
            self.tokenizer.enable_padding(direction="right", pad_id=3, pad_token="[PAD]")

            # Type of normalization we want
            self.tokenizer.normalizer = normalizers.Sequence(
                [normalizers.Lowercase()]
            )
        else:
            self.tokenizer = ByteLevelBPETokenizer()
            self.trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                               #special_tokens=special_tokens,
                                               min_frequency=2)

        # Truncation
        if self.config['enableTruncation'] == 'True':
            self.tokenizer.enable_truncation(self.config['maxLength'])

        # Post processor
        if config['task'] == 'MLM':
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single=f"[CLS]:0 $A:0 [SEP]:0",
                pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", 1),
                    ("[SEP]", 2),
                ],
            )

    def train(self):
        if self.config['tokenizerType'] == 'BPE':
            self.tokenizer.train(files=self.file_paths, vocab_size=self.config['vocabSize'], min_frequency=2,
                                 )
        else:
            self.tokenizer.train(self.file_paths, self.trainer)

    def save(self):
        if self.config['task'] == 'MLM':
            new_tokenizer = BertTokenizerFast(tokenizer_object=self.tokenizer)
        else:
            new_tokenizer = GPT2TokenizerFast(tokenizer_object=self.tokenizer)
        new_tokenizer.save_pretrained(self.tokenizerPath)
