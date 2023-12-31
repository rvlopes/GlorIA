import argparse
import json
import math

import jsonlines
from transformers import DataCollatorWithPadding, AutoTokenizer

from datasets import load_dataset
from torch.utils.data import DataLoader

import wandb
from dotenv import load_dotenv

from evaluation.glueptpt.GLUEGenDataset import GLUEGenDataset
from trainer import TrainerAccelerate


def waterdown_task(glueTaskArg):
    if "rte" in glueTaskArg:
        return "rte"
    if "mrpc" in glueTaskArg:
        return "mrpc"
    if "stsb" in glueTaskArg:
        return "stsb"
    if "wnli" in glueTaskArg:
        return "wnli"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training script')
    # Arguments passed along from checkpoint evaluation when calling finetune script - uses slurm
    parser.add_argument('-hf', dest='hfmodel', default="model", help='HugginFace model')
    parser.add_argument('-bm', dest='baseModel', default="model", help='Base model')
    parser.add_argument('-tr', dest='targetRun', default="default-run", help='Target run to derive checkpoint from')
    parser.add_argument('-gt', dest='glueTask', default="glue_rte", help='GLUEPTP Task')
    parser.add_argument('-saveBestCheckpoint', dest='saveBestCheckpoint', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Save on best checkpoint - should only be used for finetuning')
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

    # Load Glue Finetune params
    params = {
        "baseModel": args.baseModel,
        "maxLength": args.ml,
        "glue": {
            "batchSize": args.bs,
            "lr": args.lr,
            "wd": args.wd,
            "ws": args.ws,
            "maxLength": args.ml,
            "epochs": args.epochs,
            "maxSteps": -1,
            "loggingSteps": args.ls,
            "saveSteps": -1,
            "ga": args.ga,
            "ea": 1,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler
        }
    }

    # Load env variables
    if "bert" in args.hfmodel:
        load_dotenv("../../wandb_bert.env")
    else:
        load_dotenv("../../wandb_gpt.env")

    # Wandb
    wandb.login()

    # Load glue task new test split
    glueDir = "/data/rv.lopes/benchmarks/glueptpt/" + waterdown_task(args.glueTask)
    train_data = []
    with jsonlines.open(glueDir + "/" + waterdown_task(args.glueTask) + "_train_v2.json") as f: #_new_train
        for doc in f:
            train_data.append(doc)

    validation_data = []
    with jsonlines.open(glueDir + "/" + waterdown_task(args.glueTask) + "_validation_v2.json") as f:
        for doc in f:
            validation_data.append(doc)
    #validation_data = load_dataset("PORTULAN/glue-ptpt", waterdown_task(args.glueTask))['validation']

    # Deprecated
    """
    if "rte" in args.glueTask:
        glue_params = params['glue_rte']
    if "stsb" in args.glueTask:
        glue_params = params['glue_stsb']
    if "mrpc" in args.glueTask:
        glue_params = params['glue_mrpc']
    if "wnli" in args.glueTask:
        glue_params = params['glue_wnli']
    """
    glue_params = params['glue']

    # Load tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.hfmodel,
                                                     max_len=glue_params['maxLength'])

    if "gervasio" in args.baseModel:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    # Get inner bs -> actual bs
    inner_bs = glue_params['batchSize'] // glue_params['ga']

    trainDataToLoad = GLUEGenDataset(loaded_tokenizer=loaded_tokenizer,
                                     batchSize=inner_bs,
                                     data=train_data,
                                     seq_len=glue_params['maxLength'])

    evalDataToLoad = GLUEGenDataset(loaded_tokenizer=loaded_tokenizer,
                                    batchSize=inner_bs,
                                    data=validation_data,
                                    seq_len=glue_params['maxLength'])

    data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=glue_params['maxLength'])

    # Train dataloader
    train_dataloader = DataLoader(
        trainDataToLoad, collate_fn=data_collator, batch_size=inner_bs,
        shuffle=True
    )

    # Eval dataloader
    eval_dataloader = DataLoader(
        evalDataToLoad, collate_fn=data_collator, batch_size=inner_bs,
        shuffle=False
    )

    num_train_examples = trainDataToLoad.__len__()
    num_eval_examples = evalDataToLoad.__len__()

    modelTrainer = TrainerAccelerate(batchSize=glue_params['batchSize'], batchSizeEval=glue_params['batchSize'],
                                     learningRate=glue_params['lr'], weightDecay=glue_params['wd'],
                                     warmupSteps=glue_params['ws'], epochs=glue_params['epochs'],
                                     loggingSteps=glue_params['loggingSteps'], saveSteps=glue_params['saveSteps'],
                                     baseModel=params['baseModel'], wandbRun=args.targetRun, wandb=wandb,
                                     tokenizer=loaded_tokenizer, maxSteps=glue_params['maxSteps'],
                                     gradAccum=glue_params['ga'], finetune_task=args.glueTask,
                                     maxLength=glue_params['maxLength'],
                                     fp16="bf16", train_examples=num_train_examples,
                                     eval_examples=num_eval_examples,
                                     save_best_checkpoint=args.saveBestCheckpoint, seed=args.seed
                                     )

    modelTrainer.train_loop(train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            resume=False,
                            optim=glue_params['optimizer'],
                            scheduler=glue_params['scheduler'],
                            dataset=trainDataToLoad,
                            data_collator=data_collator)
