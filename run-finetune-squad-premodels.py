import argparse
import json

from transformers import DataCollatorWithPadding, AutoTokenizer

from torch.utils.data import DataLoader

import wandb
from dotenv import load_dotenv

from evaluation.squad.SquadV2Dataset import SquadV2Dataset
from trainer import TrainerAccelerate
from utils.dataset_loader import DatasetLoaderAccelerate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training script')
    # Arguments passed along from checkpoint evaluation when calling finetune script - uses slurm
    parser.add_argument('-hf', dest='hfmodel', default="model", help='HugginFace model')
    parser.add_argument('-bm', dest='baseModel', default="model", help='Base model')
    parser.add_argument('-tr', dest='targetRun', default="default-run", help='Target run to derive checkpoint from')
    parser.add_argument('-saveBestCheckpoint', dest='saveBestCheckpoint', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Save on best checkpoint - should only be used for finetuning')
    parser.add_argument('-seed', dest='seed', type=int, default=42,
                             help='Training seed', required=False)

    parser.add_argument('-squad_bs', dest='bs', type=int, default=32, help='Training batch size',
                             required=False)
    parser.add_argument('-squad_lr', dest='lr', default="1e-4", help='Learning rate',
                             required=False)
    parser.add_argument('-squad_wd', dest='wd', type=float, default=0.01, help='Weight decay',
                             required=False)
    parser.add_argument('-squad_ws', dest='ws', type=float, default=0, help='Warmup steps',
                             required=False)
    parser.add_argument('-squad_ml', dest='ml', type=int, default=512, help='Max sequence length',
                             required=False)
    parser.add_argument('-squad_e', dest='epochs', type=int, default=3, help='Training epochs',
                             required=False)
    parser.add_argument('-squad_ls', dest='ls', type=int, default=10,
                             help='Number of steps to log and eval', required=False)
    parser.add_argument('-squad_ga', dest='ga', type=int, default=8,
                             help='Number of training steps to accumulate gradient', required=False)
    parser.add_argument('-squad_scheduler', dest='scheduler', default="linear",
                             help='Scheduler for squad')
    parser.add_argument('-squad_optimizer', dest='optimizer', default="adamw",
                             help='Optimizer for squad')
    args = parser.parse_args()

    # Load SQUAD ft params
    params = {
        "baseModel": args.baseModel,
        "squad": {
            "batchSize": args.bs,
            "lr": args.lr,
            "wd": args.wd,
            "ws": args.ws,
            "maxLength": args.ml,
            "epochs": args.epochs,
            "maxSteps": -1,
            "loggingSteps": args.ls,
            "ga": args.ga,
            "version": 1,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler
        }}
    squad_params = params['squad']

    # Load env variables
    if params['baseModel'] == "BERT":
        load_dotenv("wandb_bert.env")
    else:
        load_dotenv("wandb_gpt.env")

    # Wandb
    wandb.login()

    # Get inner bs -> actual bs
    inner_bs = squad_params['batchSize'] // squad_params['ga']

    # Load tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.hfmodel,
                                                     max_len=squad_params['maxLength'])
    if "gervasio" in args.baseModel:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    squad_loader = DatasetLoaderAccelerate("squadpt1-fixed", benchmark=True)
    squad_train = squad_loader.loadDataset(streaming=False, benchmarkSplit="train")
    squad_dev = squad_loader.loadDataset(streaming=False, benchmarkSplit="dev")

    num_train_examples = squad_train.num_rows
    num_dev_examples = squad_dev.num_rows

    max_length = squad_params['maxLength']
    stride = 20


    def prepare_train_features(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = loaded_tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Start token index of the current span in the text.
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while sequence_ids[idx] == 1 and idx < len(
                    sequence_ids) - 1:  # had to add this since idx was going out of bounds due to the last token being 1
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            sp = 0
            ep = 0
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(sp)
                end_positions.append(ep)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                sp = idx - 1
                start_positions.append(sp)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                ep = idx + 1 + 1
                end_positions.append(ep)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = loaded_tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs


    squad_train_dataset = squad_train.map(prepare_train_features, batched=True,
                                              remove_columns=squad_train.column_names)

    squad_dev_dataset = squad_dev.map(preprocess_validation_examples, batched=True,
                                          remove_columns=squad_dev.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=squad_params['maxLength'])

    # Train dataloader
    train_dataloader = DataLoader(
        squad_train_dataset, collate_fn=data_collator, batch_size=inner_bs
    )

    # Eval dataloader
    squad_dev_dataset_for_model = squad_dev_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        squad_dev_dataset_for_model, collate_fn=data_collator, batch_size=inner_bs
    )

    ft_task = "squad" + str(squad_params['version'])
    modelTrainer = TrainerAccelerate(batchSize=squad_params['batchSize'], batchSizeEval=squad_params['batchSize'],
                                     learningRate=squad_params['lr'], weightDecay=squad_params['wd'],
                                     warmupSteps=squad_params['ws'], epochs=squad_params['epochs'],
                                     loggingSteps=squad_params['loggingSteps'], saveSteps=-1,
                                     baseModel=params['baseModel'], wandbRun=args.targetRun, wandb=wandb,
                                     tokenizer=loaded_tokenizer, maxSteps=squad_params['maxSteps'],
                                     gradAccum=squad_params['ga'],
                                     finetune_task=ft_task,
                                     maxLength=squad_params['maxLength'], eval_steps=-1,
                                     fp16="bf16", train_examples=num_train_examples,
                                     eval_examples=num_dev_examples, seed=args.seed,
                                     # KWARGS
                                     version=squad_params['version'],
                                     save_best_checkpoint=args.saveBestCheckpoint
                                     )

    modelTrainer.train_loop(train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            resume=False,
                            optim=squad_params['optimizer'],
                            scheduler=squad_params['scheduler'],
                            squad_dev=squad_dev,
                            squad_dev_processed=squad_dev_dataset)
