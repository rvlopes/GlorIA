from dotenv import load_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorForLanguageModeling

from accelerate_pipe.multi_dataset import MultiDataset
from accelerate_pipe.single_dataset import SingleDatasetIterable
from parsers.pretrain_parser import PreTrainParser
from tokenization.tokenizer_loader import TokenizerLoader
import wandb
import math
import os

from trainer import TrainerAccelerate

if __name__ == '__main__':
    parser_wrapper = PreTrainParser()
    parser = parser_wrapper.getParser()

    # Parse arguments
    args = parser.parse_args()
    print(args)
    args.learningRate = float(args.learningRate)
    parser_wrapper.saveFinetuneJson(args)

    # Load env variables
    if "BERT" in args.baseModel:
        load_dotenv("wandb_bert.env")
    elif "GPT2" in args.baseModel:
        load_dotenv("wandb_gpt.env")
    elif "NEO" in args.baseModel:
        load_dotenv("wandb_gptneo.env")
    else:
        load_dotenv("wandb_bloom.env")

    # Wandb
    wandb.login()

    # Load tokenizer
    tokenizerLoader = TokenizerLoader(args.tokenizer)
    loaded_tokenizer = tokenizerLoader.loadTokenizer(args.maxLength, args.baseModel)
    # Change to this if you aren't going to use a custom tokenizer
    # loaded_tokenizer = AutoTokenizer.from_pretrained("rvlopes/glorIA-1.3B")

    rank = int(os.getenv("RANK"))
    num_procs = int(os.getenv("WORLD_SIZE"))

    # Create single datasets
    ptwiki_dataset = SingleDatasetIterable("ptwiki",
                                           loaded_tokenizer=loaded_tokenizer,
                                           seq_len=args.maxLength, targetDataset="ptwikidocs-train-clean",  # -clean
                                           docs_per_file=100000,
                                           debug=args.debug,
                                           proc_rank=rank, num_procs=num_procs)  # debug=True
    arquivo_dataset = SingleDatasetIterable("arquivo",
                                            loaded_tokenizer=loaded_tokenizer,
                                            seq_len=args.maxLength, targetDataset="arquivodocs-train-clean",  # -clean
                                            docs_per_file=100000,
                                            debug=args.debug,
                                            proc_rank=rank, num_procs=num_procs)  # debug=
    clueweb_dataset = SingleDatasetIterable("clueweb",
                                            loaded_tokenizer=loaded_tokenizer,
                                            seq_len=args.maxLength, targetDataset="clueweb-large-ptpt-train",  # -ptpt
                                            docs_per_file=100000,
                                            debug=args.debug,
                                            streaming=True,
                                            proc_rank=rank, num_procs=num_procs)
    europarl_dataset = SingleDatasetIterable("europarl",
                                             loaded_tokenizer=loaded_tokenizer,
                                             seq_len=args.maxLength, targetDataset="europarldocs-train",
                                             docs_per_file=100000,
                                             debug=args.debug,
                                             proc_rank=rank, num_procs=num_procs)
    oscar_dataset = SingleDatasetIterable("oscar",
                                          loaded_tokenizer=loaded_tokenizer,
                                          seq_len=args.maxLength, targetDataset="oscar-ptpt-train-clean",
                                          docs_per_file=100000,
                                          debug=args.debug,
                                          proc_rank=rank, num_procs=num_procs)
    opensubtitles_dataset = SingleDatasetIterable("subtitles",
                                                  loaded_tokenizer=loaded_tokenizer,
                                                  seq_len=args.maxLength, targetDataset="opensubtitles-filtered",
                                                  docs_per_file=100000,
                                                  debug=args.debug,
                                                  proc_rank=rank, num_procs=num_procs)

    # Prepare and create multidataset with sampling
    # [0.3, 0.53, 0.17] clean | [0.17, 0.28, 0.55] no clean | [0.37, 0.63]  # ptwiki + arquivo
    # weights = [0.3, 0.53, 0.17] # [0.3, 0.2, 0.1, 0.4]  # custom weights
    # ND WEIGHTS [0.1, 0.1, 0.15, 0.05, 0.6]
    # ND2 WEIGHTS [0.2, 0.1, 0.15, 0.05, 0.4]
    # OD WEIGHTS [0.15, 0.1, 0.5, 0.05, 0.2]
    # PTWIKI,   ARQUIVO, CLUEWEB, EUROPARL, OSCAR, OPENSUBTITLES
    weights = [0.1, 0.08, 0.63, 0.06, 0.08, 0.06]
    infiniteMode = True if args.maxSteps > 0 else False
    datasets = [ptwiki_dataset, arquivo_dataset, clueweb_dataset,
                europarl_dataset, oscar_dataset, opensubtitles_dataset]
    multi_dataset = MultiDataset(datasets=datasets,
                                 weights=weights,
                                 infiniteMode=infiniteMode)

    #inner_bs = args.batchSize // args.gradAccum

    # Dataloaders receive a datacollator for MLM when training BERT, instead of the default
    if args.baseModel == "BERT":
        data_collator = DataCollatorForLanguageModeling(loaded_tokenizer, mlm=True)
    else:
        data_collator = DataCollatorForLanguageModeling(loaded_tokenizer, mlm=False)

    # DataLoaders creation:
    # Train dataloader
    num_examples = len(multi_dataset)
    print("Total examples: " + str(num_examples))

    # Instantiate trainer class
    modelTrainer = TrainerAccelerate(batchSize=args.batchSize, learningRate=args.learningRate,
                                     weightDecay=args.weightDecay, warmupSteps=args.warmupSteps,
                                     epochs=args.epochs, loggingSteps=args.loggingSteps,
                                     saveSteps=args.saveSteps, baseModel=args.baseModel,
                                     wandbRun=args.wandb, wandb=wandb, tokenizer=loaded_tokenizer,
                                     maxSteps=args.maxSteps, fp16=args.fp16, gradAccum=args.gradAccum,
                                     eval_steps=args.saveSteps, train_examples=num_examples,
                                     deepspeed=args.deepspeed)

    modelTrainer.train_loop(resume=args.resume,
                            optim=args.optimizer,
                            scheduler=args.scheduler,
                            dataset=multi_dataset,
                            data_collator=data_collator,
                            weights=weights.copy(),
                            resume_checkpoint=args.resumeCheckpoint,
                            hard_rr=args.hardRestarts)
