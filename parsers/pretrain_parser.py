import argparse
import os
import json
from pathlib import Path


class PreTrainParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Model training script')

        # PRE TRAIN PARAMS
        self.parser.add_argument('-b', dest='batchSize', type=int, default=32, help='Training batch size')
        self.parser.add_argument('-lr', dest='learningRate', default="1e-4", help='Learning rate')
        self.parser.add_argument('-wd', dest='weightDecay', type=float, default=0, help='Weight decay')
        self.parser.add_argument('-ws', dest='warmupSteps', type=float, default=0, help='Warmup steps')
        self.parser.add_argument('-ml', dest='maxLength', type=int, default=128, help='Max sequence length')
        self.parser.add_argument('-e', dest='epochs', type=int, default=1, help='Training epochs')
        self.parser.add_argument('-ms', dest='maxSteps', type=int, default=-1, help='Maximum number of steps epochs')
        self.parser.add_argument('-d', dest='dataset', default="ptwikicleanv2", help='Dataset to use')
        self.parser.add_argument('-t', dest='tokenizer', default="gptuga-tk-v1", help='Tokenizer to use')
        self.parser.add_argument('-m', dest='baseModel', default="GPT2-124", help='Base model to use')
        self.parser.add_argument('-ls', dest='loggingSteps', type=int, default=50,
                                 help='Number of steps to log and eval')
        self.parser.add_argument('-ss', dest='saveSteps', type=int, default=-1, help='Number of steps to save')
        self.parser.add_argument('-wandb', dest='wandb', default="no-name", help='Wandb run name')
        self.parser.add_argument('-wandb_env', dest='wandb_env', default="wandb_gpt.env", help='Wandb env file')
        self.parser.add_argument('-ga', dest='gradAccum', type=int, default=1,
                                 help='Number of training steps to accumulate '
                                      'gradient')
        self.parser.add_argument('-fp16', dest='fp16', default=False,
                                 help='FP16 mixed precision training')
        self.parser.add_argument('-scheduler', dest='scheduler', default="cosine", help='Scheduler')
        self.parser.add_argument('-optimizer', dest='optimizer', default="adamw", help='Optimizer')
        self.parser.add_argument('-resume', dest='resume', default=False, action=argparse.BooleanOptionalAction,
                                 help='Resume pre-training')
        self.parser.add_argument('-checkpoint', dest='resumeCheckpoint', default=None,
                                 help='Checkpoint to resume training from')
        self.parser.add_argument('-hrr', dest='hardRestarts', type=int, default=0,
                                 help='Number of hard restarts to perform for scheduler')
        self.parser.add_argument('-seed', dest='seed', type=int, default=42,
                                 help='Training seed', required=False)

        # OTHER PARAMS
        self.parser.add_argument('-debug', dest='debug', default=False, action=argparse.BooleanOptionalAction,
                                 help='Debug mode - affects sizes of data to be loaded')
        self.parser.add_argument('-deepspeed', dest='deepspeed', default=False, action=argparse.BooleanOptionalAction,
                                 help='Use deepspeed')

        # FINETUNE PARAMS ASSIN2
        self.parser.add_argument('-assin_bs', dest='assin_batchSize', type=int, default=32, help='Training batch size',
                                 required=False)
        self.parser.add_argument('-assin_lr', dest='assin_learningRate', default="1e-4", help='Learning rate',
                                 required=False)
        self.parser.add_argument('-assin_wd', dest='assin_weightDecay', type=float, default=0.01, help='Weight decay',
                                 required=False)
        self.parser.add_argument('-assin_ws', dest='assin_warmupSteps', type=float, default=0, help='Warmup steps',
                                 required=False)
        self.parser.add_argument('-assin_ml', dest='assin_maxLength', type=int, default=128, help='Max sequence length',
                                 required=False)
        self.parser.add_argument('-assin_e', dest='assin_epochs', type=int, default=1, help='Training epochs',
                                 required=False)
        self.parser.add_argument('-assin_ms', dest='assin_maxSteps', type=int, default=-1,
                                 help='Maximum number of steps epochs', required=False)
        self.parser.add_argument('-assin_ls', dest='assin_loggingSteps', type=int, default=5000,
                                 help='Number of steps to log and eval', required=False)
        self.parser.add_argument('-assin_ss', dest='assin_saveSteps', type=int, default=10000,
                                 help='Number of steps to save',
                                 required=False)
        self.parser.add_argument('-assin_ga', dest='assin_gradAccum', type=int, default="1",
                                 help='Number of training steps to accumulate gradient', required=False)
        self.parser.add_argument('-assin_version', dest='assin_version', type=int, default=2,
                                 help='Which version of ASSIN to use', required=False)
        self.parser.add_argument('-assin_scheduler', dest='assin_scheduler', default="cosine",
                                 help='Scheduler for ASSIN tasks')
        self.parser.add_argument('-assin_optimizer', dest='assin_optimizer', default="radam",
                                 help='Optimizer for ASSIN tasks')

        # FINETUNE PARAMS SQUAD
        self.parser.add_argument('-squad_bs', dest='squad_batchSize', type=int, default=32, help='Training batch size',
                                 required=False)
        self.parser.add_argument('-squad_lr', dest='squad_learningRate', default="1e-4", help='Learning rate',
                                 required=False)
        self.parser.add_argument('-squad_wd', dest='squad_weightDecay', type=float, default=0.01, help='Weight decay',
                                 required=False)
        self.parser.add_argument('-squad_ws', dest='squad_warmupSteps', type=float, default=0, help='Warmup steps',
                                 required=False)
        self.parser.add_argument('-squad_ml', dest='squad_maxLength', type=int, default=512, help='Max sequence length',
                                 required=False)
        self.parser.add_argument('-squad_e', dest='squad_epochs', type=int, default=1, help='Training epochs',
                                 required=False)
        self.parser.add_argument('-squad_ms', dest='squad_maxSteps', type=int, default=-1,
                                 help='Maximum number of steps epochs', required=False)
        self.parser.add_argument('-squad_ls', dest='squad_loggingSteps', type=int, default=5000,
                                 help='Number of steps to log and eval', required=False)
        self.parser.add_argument('-squad_ss', dest='squad_saveSteps', type=int, default=-1,
                                 help='Number of steps to save',
                                 required=False)
        self.parser.add_argument('-squad_ga', dest='squad_gradAccum', type=int, default=1,
                                 help='Number of training steps to accumulate gradient', required=False)
        self.parser.add_argument('-squad_version', dest='squad_version', type=int, default=1,
                                 help='Which version of SQUAD to use', required=False)
        self.parser.add_argument('-squad_scheduler', dest='squad_scheduler', default="cosine",
                                 help='Scheduler for squad')
        self.parser.add_argument('-squad_optimizer', dest='squad_optimizer', default="radam",
                                 help='Optimizer for squad')

        # FINETUNE PARAMS GLUE TASKS
        self.parser.add_argument('-glue_bs', dest='glue_batchSize', type=int, default=32, help='Training batch size',
                                 required=False)
        self.parser.add_argument('-glue_lr', dest='glue_learningRate', default="1e-4", help='Learning rate',
                                 required=False)
        self.parser.add_argument('-glue_wd', dest='glue_weightDecay', type=float, default=0.01, help='Weight decay',
                                 required=False)
        self.parser.add_argument('-glue_ws', dest='glue_warmupSteps', type=float, default=0, help='Warmup steps',
                                 required=False)
        self.parser.add_argument('-glue_ml', dest='glue_maxLength', type=int, default=512, help='Max sequence length',
                                 required=False)
        self.parser.add_argument('-glue_e', dest='glue_epochs', type=int, default=1, help='Training epochs',
                                 required=False)
        self.parser.add_argument('-glue_ms', dest='glue_maxSteps', type=int, default=-1,
                                 help='Maximum number of steps epochs', required=False)
        self.parser.add_argument('-glue_ls', dest='glue_loggingSteps', type=int, default=5000,
                                 help='Number of steps to log and eval', required=False)
        self.parser.add_argument('-glue_ss', dest='glue_saveSteps', type=int, default=-1,
                                 help='Number of steps to save',
                                 required=False)
        self.parser.add_argument('-glue_ga', dest='glue_gradAccum', type=int, default=1,
                                 help='Number of training steps to accumulate gradient', required=False)
        self.parser.add_argument('-glue_scheduler', dest='glue_scheduler', default="cosine",
                                 help='Scheduler for squad')
        self.parser.add_argument('-glue_optimizer', dest='glue_optimizer', default="adamw",
                                 help='Optimizer for squad')

    def getParser(self):
        return self.parser

    # Received parsed parser
    def saveFinetuneJson(self, args):
        ft_params = {}
        # SAVE ASSIN FINETUNE VALUES
        ft_params['assin'] = {
            "batchSize": args.assin_batchSize,
            "lr": args.assin_learningRate,
            "wd": args.assin_weightDecay,
            "ws": args.assin_warmupSteps,
            "maxLength": args.assin_maxLength,
            "epochs": args.assin_epochs,
            "maxSteps": args.maxSteps,
            "loggingSteps": args.assin_loggingSteps,
            "saveSteps": args.assin_saveSteps,
            "ga": args.assin_gradAccum,
            "version": args.assin_version,
            "optimizer": args.assin_optimizer,
            "scheduler": args.assin_scheduler
        }

        # SAVE SQUAD FT PARAMS
        ft_params['squad'] = {
            "batchSize": args.squad_batchSize,
            "lr": args.squad_learningRate,
            "wd": args.squad_weightDecay,
            "ws": args.squad_warmupSteps,
            "maxLength": args.squad_maxLength,
            "epochs": args.squad_epochs,
            "maxSteps": args.squad_maxSteps,
            "loggingSteps": args.squad_loggingSteps,
            "saveSteps": args.squad_saveSteps,
            "ga": args.squad_gradAccum,
            "version": args.squad_version,
            "optimizer": args.squad_optimizer,
            "scheduler": args.squad_scheduler
        }

        # SAVE GLUE FT PARAMS
        ft_params['glue'] = {
            "batchSize": args.glue_batchSize,
            "lr": args.glue_learningRate,
            "wd": args.glue_weightDecay,
            "ws": args.glue_warmupSteps,
            "maxLength": args.glue_maxLength,
            "epochs": args.glue_epochs,
            "maxSteps": args.glue_maxSteps,
            "loggingSteps": args.glue_loggingSteps,
            "saveSteps": args.glue_saveSteps,
            "ga": args.glue_gradAccum,
            "optimizer": args.glue_optimizer,
            "scheduler": args.glue_scheduler
        }

        # GLOBAL PARAMS
        ft_params["tokenizer"] = args.tokenizer
        ft_params["baseModel"] = args.baseModel
        ft_params["dataset"] = args.dataset
        ft_params["maxLength"] = args.maxLength
        ft_params['fp16'] = args.fp16
        ft_params['modelName'] = args.wandb

        # Save params in model directory
        modelSaveDir = "/data/rv.lopes/models/" + args.wandb
        if not os.path.exists(modelSaveDir):
            # os.mkdir(modelSaveDir, exist_ok=True)
            Path(modelSaveDir).mkdir(exist_ok=True)
        finetuneParamsFile = modelSaveDir + "/params.json"
        if not os.path.exists(finetuneParamsFile):
            json_object = json.dumps(ft_params, indent=4)
            with open(finetuneParamsFile, "w") as outfile:
                outfile.write(json_object)
