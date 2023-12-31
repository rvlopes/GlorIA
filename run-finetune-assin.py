import argparse
import json
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv
from evaluation.assin.Assin2Processor import preprocess_assin2
from evaluation.assin.AssinDataset import AssinDataset
from tokenization.tokenizer_loader import TokenizerLoader
from trainer import TrainerAccelerate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training script')
    # Arguments passed along from checkpoint evaluation when calling finetune script - uses slurm
    parser.add_argument('-c', dest='checkpoint', default="model-checkpoint", help='Checkpoint folder name')
    parser.add_argument('-tr', dest='targetRun', default="default-run", help='Target run to derive checkpoint from')
    parser.add_argument('-at', dest='assinTask', default="assin_similarity", help='ASSIN Task')
    parser.add_argument('-saveBestCheckpoint', dest='saveBestCheckpoint', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Save on best checkpoint - should only be used for finetuning')
    parser.add_argument('-v', dest='version', type=int, default=2,
                        help='Which version of ASSIN to use', required=False)
    parser.add_argument('-seed', dest='seed', type=int, default=42,
                             help='Training seed', required=False)
    parser.add_argument('-bs', dest='bs', type=int, default=32, help='Training batch size',
                             required=False)
    parser.add_argument('-lr', dest='lr', default="1e-5", help='Learning rate',
                             required=False)
    parser.add_argument('-wd', dest='wd', type=float, default=0.01, help='Weight decay',
                             required=False)
    parser.add_argument('-ws', dest='ws', type=float, default=0, help='Warmup steps',
                             required=False)
    parser.add_argument('-ml', dest='ml', type=int, default=128, help='Max sequence length',
                             required=False)
    parser.add_argument('-e', dest='epochs', type=int, default=5, help='Training epochs',
                             required=False)
    parser.add_argument('-ls', dest='ls', type=int, default=10,
                             help='Number of steps to log and eval', required=False)
    parser.add_argument('-ga', dest='ga', type=int, default=2,
                             help='Number of training steps to accumulate gradient', required=False)
    parser.add_argument('-scheduler', dest='scheduler', default="linear",
                             help='Scheduler')
    parser.add_argument('-optimizer', dest='optimizer', default="adamw",
                             help='Optimizer')
    args = parser.parse_args()

    # Load Assin Finetune params specificed in JSON
    model_dir = "/data/rv.lopes/models/" + args.targetRun
    finetuneParamsFile = model_dir + "/params.json"
    with open(finetuneParamsFile, 'r') as openfile:
        params = json.load(openfile)
    assin_params = params['assin']  # TODO DONT FORGET TO CHANGE THIS

    # Load env variables
    if params['baseModel'] == "BERT":
        load_dotenv("/env_files/wandb_bert.env")
    else:
        load_dotenv("/env_files/wandb_gpt.env")

    # Wandb
    wandb.login()

    # Load assin - version 1 comes from huggingface, while the second needs to
    # be locally stored and loaded, since the huggingface version is not correct
    if args.version == 1:
        # Load assin1 https://huggingface.co/datasets/assin/viewer/full/train
        data = preprocess_assin2("assin1_pickled", version=1, subset="ptpt")  # load_dataset("assin", name="ptpt")
        assin_params = params['assin1']
    else:
        # Load assin2
        data = preprocess_assin2("assin2_pickled")
        assin_params = params['assin']

    # Update params - long story on why i did this
    assin_params['batchSize'] = args.bs
    assin_params['lr'] = args.lr
    assin_params['ws'] = args.wd
    assin_params['ws'] = args.ws
    assin_params['maxLength'] = args.ml
    assin_params['epochs'] = args.epochs
    assin_params['loggingSteps'] = args.ls
    assin_params['ga'] = args.ga
    assin_params['optimizer'] = args.optimizer
    assin_params['scheduler'] = args.scheduler

    # Load tokenizer
    tokenizerLoader = TokenizerLoader(params['tokenizer'])
    loaded_tokenizer = tokenizerLoader.loadTokenizer(assin_params['maxLength'], params['baseModel'])

    # Get inner bs -> actual bs
    inner_bs = assin_params['batchSize'] // assin_params['ga']

    if args.assinTask == "assin":
        trainDataToLoad = AssinDataset(loaded_tokenizer=loaded_tokenizer,
                                       batchSize=inner_bs,  # assin_params['batchSize'],
                                       data=data['train'],
                                       seq_len=assin_params['maxLength'],
                                       categoric=False,
                                       multitask=True)

        evalDataToLoad = AssinDataset(loaded_tokenizer=loaded_tokenizer,
                                      batchSize=inner_bs,  # assin_params['batchSize'],
                                      data=data['validation'],
                                      seq_len=assin_params['maxLength'],
                                      categoric=False,
                                      multitask=True)
    else:
        categoric = True
        if args.assinTask == "assin_similarity":
            categoric = False

        trainDataToLoad = AssinDataset(loaded_tokenizer=loaded_tokenizer,
                                       batchSize=inner_bs,  # assin_params['batchSize'],
                                       data=data['train'],
                                       seq_len=assin_params['maxLength'],
                                       categoric=categoric)

        evalDataToLoad = AssinDataset(loaded_tokenizer=loaded_tokenizer,
                                      batchSize=inner_bs,  # assin_params['batchSize'],
                                      data=data['validation'],
                                      seq_len=assin_params['maxLength'],
                                      categoric=categoric)

    # data_collator = DefaultDataCollator()
    data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=assin_params['maxLength'])

    # Train dataloader
    train_dataloader = DataLoader(
        trainDataToLoad, collate_fn=data_collator, batch_size=inner_bs,  # assin_params['batchSize'],
        shuffle=True
    )

    # Eval dataloader
    eval_dataloader = DataLoader(
        evalDataToLoad, collate_fn=data_collator, batch_size=inner_bs,  # assin_params['batchSize'],
        shuffle=False
    )

    num_train_examples = trainDataToLoad.__len__()
    num_eval_examples = evalDataToLoad.__len__()

    modelTrainer = TrainerAccelerate(batchSize=assin_params['batchSize'], batchSizeEval=assin_params['batchSize'],
                                     learningRate=assin_params['lr'], weightDecay=assin_params['wd'],
                                     warmupSteps=assin_params['ws'], epochs=assin_params['epochs'],
                                     loggingSteps=assin_params['loggingSteps'], saveSteps=assin_params['saveSteps'],
                                     baseModel=params['baseModel'], wandbRun=args.targetRun, wandb=wandb,
                                     tokenizer=loaded_tokenizer, maxSteps=assin_params['maxSteps'],
                                     gradAccum=assin_params['ga'],finetune_task=args.assinTask,
                                     checkpoint=args.checkpoint,maxLength=assin_params['maxLength'],
                                     fp16=params['fp16'], train_examples=num_train_examples,
                                     eval_examples=num_eval_examples, seed=args.seed,
                                     save_best_checkpoint=args.saveBestCheckpoint,
                                     # KWARGS
                                     version=args.version
                                     )

    modelTrainer.train_loop(train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            resume=False,
                            optim=assin_params['optimizer'],
                            scheduler=assin_params['scheduler'],
                            dataset=trainDataToLoad,
                            data_collator=data_collator)
